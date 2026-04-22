import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] 
RAFT_PATH = ROOT / "external" / "RAFT-Stereo"

if str(RAFT_PATH) not in sys.path:
    sys.path.append(str(RAFT_PATH))
    sys.path.append(str(RAFT_PATH / "core"))

from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.Geometry.triangulation.triangulator import Triangulator
import cv2
from tqdm import tqdm
import csv
from datetime import datetime

class RAFTSTEREOInferencer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # RAFT-Stereo internal config
        class Args:
            restore_ckpt = checkpoint_path
            mixed_precision = True
            valid_iters = 32
            hidden_dims = [128]*3
            context_dims = [128]*3
            corr_implementation = "alt" 
            shared_backbone = False
            corr_levels = 4
            corr_radius = 4
            n_downsample = 2
            slow_fast_gru = False
            n_gru_layers = 3
            context_norm = 'batch'

        # Initialize model
        self.model = torch.nn.DataParallel(RAFTStereo(Args()))
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = self.model.module.to(device).eval()

    def get_disparity(self, rect_l, rect_r):
        """
        Takes rectified images (OpenCV/BGR) and returns a dense disparity map.
        """
        # BGR to RGB (RAFT was trained on RGB)
        img_l = rect_l[:, :, ::-1].copy()
        img_r = rect_r[:, :, ::-1].copy()

        # Convert to Tensors [1, 3, H, W] (Scaling 0-255 as per demo.py)
        t_l = torch.from_numpy(img_l).permute(2, 0, 1).float()[None].to(self.device)
        t_r = torch.from_numpy(img_r).permute(2, 0, 1).float()[None].to(self.device)

        # Pad to multiples of 32
        padder = InputPadder(t_l.shape, divis_by=32)
        t_l, t_r = padder.pad(t_l, t_r)

        # Inference
        with torch.no_grad():
            _, flow_up = self.model(t_l, t_r, iters=32, test_mode=True)
            disparity = padder.unpad(flow_up).squeeze().cpu().numpy()

        # RAFT-Stereo predicts flow from Left to Right (negative values)
        # We negate it to get positive disparity values for depth projection
        return -disparity

    def save_output(self, disp, lrc_error, lrc_mask, output_dir, file_stem, save_png=False):
        base_path = Path(output_dir)
        
        #Save compressed numerical data
        data_folder = base_path / "compressed_data"
        data_folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_folder / f"{file_stem}.npz", 
            disparity=disp, 
            lrc_error=lrc_error, 
            lrc_mask=lrc_mask
        )

        # Optional Visualization
        if save_png:
            png_folder = base_path / "plots_disparities"
            png_folder.mkdir(parents=True, exist_ok=True)
            plt.imsave(png_folder / f"{file_stem}_disp.png", disp, cmap='jet')
        
      
        
    # def run_batch_inference(self, left_img_root, right_img_root, zip_root, output_dir, video_ids, img_shape):
    #     for vid in tqdm(video_ids, desc='Overall Progress', position=0, unit='video'):
    #         left_vid_path= os.path.join(left_img_root,vid)
    #         right_vid_path= os.path.join(right_img_root,vid)
    #         zip_path = os.path.join(zip_root,vid +".zip")
    #         triangulator = Triangulator()
    #         triangulator.load_calibration(zip_path)
    #         left_frames =  sorted(os.listdir(left_vid_path))
    #         right_frames =  sorted(os.listdir(right_vid_path))
    #         h, w = img_shape
    #         lmap1, lmap2, rmap1, rmap2, q= triangulator.get_rectification_maps(img_size=(h,w), mode="conventional")
    #         for l_frame, r_frame in tqdm(zip(left_frames, right_frames), desc=f'Video {vid}', position=1, leave=False, total=len(left_frames),unit='frame'):
    #             assert l_frame.replace("left", "right") == r_frame, "the frams are not correspondant to each other"
    #             l_frame_path= os.path.join(left_vid_path,l_frame)
    #             r_frame_path= os.path.join(right_vid_path,r_frame)
    #             img_l = cv2.imread(l_frame_path)
    #             img_r= cv2.imread(r_frame_path)
    #             rect_l, rect_r = triangulator.rectify_images(img_l, img_r,lmap1, lmap2, rmap1, rmap2, "conventional")
    #             disparity = self.get_disparity(rect_l,rect_r)
    #             file_stem = os.path.basename(l_frame.replace("_left", "")).split('.')[0]
    #             self.save_output(disparity, output_dir, file_stem)
    #             torch.cuda.empty_cache()

    def run_batch_inference(self, left_img_root, right_img_root, zip_root, output_dir, video_ids, img_shape, lrc_threshold= 1 ,save_visuals=False):
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV log
        csv_path = output_path / f"lrc_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_headers = ['video_id', 'frame', 'lrc_mean', 'lrc_std', 'lrc_max', 'lrc_min', 'consistency_rate']
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

        for vid in tqdm(video_ids, desc='Overall Progress', position=0, unit='video'):
            # Path and Triangulator setup
            left_vid_path = os.path.join(left_img_root, vid)
            right_vid_path = os.path.join(right_img_root, vid)
            zip_path = os.path.join(zip_root, vid + ".zip")
            
            triangulator = Triangulator()
            triangulator.load_calibration(zip_path)
            h, w = img_shape
            lmap1, lmap2, rmap1, rmap2, q = triangulator.get_rectification_maps(img_size=(h,w), mode="conventional")
            
            left_frames = sorted(os.listdir(left_vid_path))
            right_frames = sorted(os.listdir(right_vid_path))

            for l_frame, r_frame in tqdm(zip(left_frames, right_frames), desc=f'Video {vid}', position=1, leave=False, total=len(left_frames)):
                #Load and Rectify
                img_l = cv2.imread(os.path.join(left_vid_path, l_frame))
                img_r = cv2.imread(os.path.join(right_vid_path, r_frame))
                rect_l, rect_r = triangulator.rectify_images(img_l, img_r, lmap1, lmap2, rmap1, rmap2, "conventional")
                
                #Bidirectional Inference
                disp_l, disp_r = self.get_bidirectional_disparity(rect_l, rect_r)
                
                #Compute LRC metrics
                lrc_mask, lrc_error = self.compute_lrc_mask(disp_l, disp_r, threshold=lrc_threshold)
                
                stats = {
                    'video_id': vid,
                    'frame': os.path.basename(l_frame).split('.')[0],
                    'lrc_mean': round(float(np.mean(lrc_error)), 4),
                    'lrc_std': round(float(np.std(lrc_error)), 4),
                    'lrc_max': round(float(np.max(lrc_error)), 4),
                    'lrc_min': round(float(np.min(lrc_error)), 4),
                    'consistency_rate': round(float(np.mean(lrc_mask) * 100), 2)
                }
                
                # Append to CSV 
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    writer.writerow(stats)
                
                # Save Arrays
                file_stem = os.path.basename(l_frame.replace("_left", "")).split('.')[0]
                self.save_output(disp_l, lrc_error, lrc_mask, output_path / vid, file_stem, save_png=save_visuals)
  
                torch.cuda.empty_cache()

   

    def get_bidirectional_disparity(self, rect_l, rect_r):
        # Standard Left-to-Right
        disp_l = self.get_disparity(rect_l, rect_r)
        
        # Right-to-Left 
        # Flip images horizontally
        rect_l_flipped = cv2.flip(rect_l, 1)
        rect_r_flipped = cv2.flip(rect_r, 1)
        
        # Run inference on flipped images
        disp_r_flipped = self.get_disparity(rect_r_flipped, rect_l_flipped)
        
        # Flip the resulting disparity map back
        disp_r = cv2.flip(disp_r_flipped, 1)
        
        return disp_l, disp_r

    def compute_lrc_mask(self, disp_l, disp_r, threshold=1.0):
        """
        Performs the Left-Right Consistency check.
        Returns a mask where 1 is consistent and 0 is inconsistent.
        """
        h, w = disp_l.shape
        # Create a grid of coordinates
        u_coords = np.tile(np.arange(w), (h, 1))
        
        # Calculate where each pixel in Left landed in Right
        # target_u is the coordinate in the right image
        target_u = u_coords - disp_l
        target_u = np.clip(target_u, 0, w - 1).astype(np.float32)
        
        # Re-map the Right-to-Left disparity back to the Left perspective
        # We need to know what the right disparity says at the point we matched to
        v_coords = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
        
        # Use cv2.remap to find the disparity value in disp_r at the matched locations
        projected_disp_r = cv2.remap(disp_r, target_u, v_coords, cv2.INTER_LINEAR)
        
        # The Check: |disp_l - projected_disp_r|
        diff = np.abs(disp_l - projected_disp_r)
        mask = (diff < threshold).astype(np.float32)
        
        return mask, diff
    

    
  
