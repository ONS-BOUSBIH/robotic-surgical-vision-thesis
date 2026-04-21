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

    def save_output(self, disparity, output_dir, file_stem):
        """
        Saves the raw disparity array and the png.
        """
        base_path = Path(output_dir)

        # Save raw disparities
        npy_folder = base_path / "npy_disparities"
        npy_folder.mkdir(parents=True, exist_ok=True)
        npy_filename = npy_folder / f"{file_stem}.npy"
        np.save(npy_filename, disparity)

        # Save png maps
        png_folder = base_path / "plots_disparities"
        png_folder.mkdir(parents=True, exist_ok=True)
        png_filename = png_folder / f"{file_stem}.png"
        
        plt.imsave(png_filename, disparity, cmap='jet')
        
      
        
    def run_inference(self, left_img_root, right_img_root, zip_root, output_dir, video_ids, img_shape):
        for vid in tqdm(video_ids, desc='Overall Progress', position=0, unit='video'):
            left_vid_path= os.path.join(left_img_root,vid)
            right_vid_path= os.path.join(right_img_root,vid)
            zip_path = os.path.join(zip_root,vid +".zip")
            triangulator = Triangulator()
            triangulator.load_calibration(zip_path)
            left_frames =  sorted(os.listdir(left_vid_path))
            right_frames =  sorted(os.listdir(right_vid_path))
            h, w = img_shape
            lmap1, lmap2, rmap1, rmap2, q= triangulator.get_rectification_maps(img_size=(h,w), mode="conventional")
            for l_frame, r_frame in tqdm(zip(left_frames, right_frames), desc=f'Video {vid}', position=1, leave=False, total=len(left_frames),unit='frame'):
                assert l_frame.replace("left", "right") == r_frame, "the frams are not correspondant to each other"
                l_frame_path= os.path.join(left_vid_path,l_frame)
                r_frame_path= os.path.join(right_vid_path,r_frame)
                img_l = cv2.imread(l_frame_path)
                img_r= cv2.imread(r_frame_path)
                rect_l, rect_r = triangulator.rectify_images(img_l, img_r,lmap1, lmap2, rmap1, rmap2, "conventional")
                disparity = self.get_disparity(rect_l,rect_r)
                file_stem = os.path.basename(l_frame.replace("_left", "")).split('.')[0]
                self.save_output(disparity, output_dir, file_stem)
                torch.cuda.empty_cache()


    

    
  
