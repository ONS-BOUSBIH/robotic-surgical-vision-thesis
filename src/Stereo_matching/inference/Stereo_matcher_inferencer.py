import cv2
import numpy as np
import os
import csv
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from src.Geometry.triangulation.triangulator import Triangulator
from skimage.metrics import structural_similarity as ssim

class StereoMatcherInferencer:
    def __init__(self, device='cuda'):
        self.device = device

    def get_disparity(self, rect_l, rect_r):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def get_bidirectional_disparity(self, rect_l, rect_r):
        # Standard Left-to-Right
        disp_l = self.get_disparity(rect_l, rect_r)
        
        # Right-to-Left via Flip Method
        rect_l_flipped = cv2.flip(rect_l, 1)
        rect_r_flipped = cv2.flip(rect_r, 1)
        
        disp_r_flipped = self.get_disparity(rect_r_flipped, rect_l_flipped)
        disp_r = cv2.flip(disp_r_flipped, 1)
        
        return disp_l, disp_r

    def compute_lrc_mask(self, disp_l, disp_r, threshold=1.0):
        h, w = disp_l.shape
        u_coords = np.tile(np.arange(w), (h, 1))
        target_u = np.clip(u_coords - disp_l, 0, w - 1).astype(np.float32)
        v_coords = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
        
        projected_disp_r = cv2.remap(disp_r, target_u, v_coords, cv2.INTER_LINEAR)
        
        diff = np.abs(disp_l - projected_disp_r)
        mask = (diff < threshold).astype(np.float32)
        return mask, diff
  
    def compute_reprojection_ssim(self, img_l, img_r, disp_l):
        
        #Warp the Right image to the Left perspective using disp_l
        h, w = disp_l.shape
        u_coords = np.tile(np.arange(w), (h, 1))
        v_coords = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
        
        target_u = np.clip(u_coords - disp_l, 0, w - 1).astype(np.float32)
        
        # Warp img_r
        warped_r = cv2.remap(img_r, target_u, v_coords, cv2.INTER_LINEAR)
        
        # Calculate SSIM between Original Left and Warped Right
        # score, diff_map = ssim(img_l, warped_r, full=True, multichannel=True)
        score = ssim(img_l, warped_r, channel_axis=2, data_range=255)
        
        return score

    def save_output(self, disp, lrc_error, lrc_mask, output_dir, file_stem, save_png=False):
        base_path = Path(output_dir)
        data_folder = base_path / "compressed_data"
        data_folder.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            data_folder / f"{file_stem}.npz", 
            disparity=disp, 
            lrc_error=lrc_error, 
            lrc_mask=lrc_mask
        )

        if save_png:
            png_folder = base_path / "plots_disparities"
            png_folder.mkdir(parents=True, exist_ok=True)
            plt.imsave(png_folder / f"{file_stem}_disp.png", disp, cmap='jet')

    def run_batch_inference(self, left_img_root, right_img_root, zip_root, output_dir, video_ids, img_shape, lrc_threshold=1, save_visuals=False):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_path / f"lrc_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_headers = ['video_id', 'frame', 'lrc_mean', 'lrc_std', 'lrc_max', 'lrc_min', 'consistency_rate','ssim_score','lrc_median']
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

        for vid in tqdm(video_ids, desc='Overall Progress'):
            # Setup paths and triangulator
            left_vid_path = os.path.join(left_img_root, vid)
            right_vid_path = os.path.join(right_img_root, vid)
            triangulator = Triangulator()
            triangulator.load_calibration(os.path.join(zip_root, vid + ".zip"))
            
            lmap1, lmap2, rmap1, rmap2, _ = triangulator.get_rectification_maps(img_size=img_shape, mode="conventional")
            
            left_frames = sorted(os.listdir(left_vid_path))
            right_frames = sorted(os.listdir(right_vid_path))

            for l_f, r_f in tqdm(zip(left_frames, right_frames), desc=f'Vid {vid}', total=len(left_frames), leave=False):
                img_l = cv2.imread(os.path.join(left_vid_path, l_f))
                img_r = cv2.imread(os.path.join(right_vid_path, r_f))
                rect_l, rect_r = triangulator.rectify_images(img_l, img_r, lmap1, lmap2, rmap1, rmap2, "conventional")
                
                disp_l, disp_r = self.get_bidirectional_disparity(rect_l, rect_r)
                lrc_mask, lrc_error = self.compute_lrc_mask(disp_l, disp_r, threshold=lrc_threshold)
                ssim_score = self.compute_reprojection_ssim(rect_l,rect_r,disp_l)
                # Logging and Saving
                stats = {
                    'video_id': vid,
                    'frame': Path(l_f).stem,
                    'lrc_mean': round(float(np.mean(lrc_error)), 4),
                    'lrc_std': round(float(np.std(lrc_error)), 4),
                    'lrc_max': round(float(np.max(lrc_error)), 4),
                    'lrc_min': round(float(np.min(lrc_error)), 4),
                    'consistency_rate': round(float(np.mean(lrc_mask) * 100), 2),
                    'ssim_score': round(ssim_score, 4),
                    'lrc_median': round(float(np.median(lrc_error)), 4)
                }
                
                with open(csv_path, mode='a', newline='') as f:
                    csv.DictWriter(f, fieldnames=csv_headers).writerow(stats)
                
                file_stem = Path(l_f).stem.replace("_left", "")
                self.save_output(disp_l, lrc_error, lrc_mask, output_path / vid, file_stem, save_png=save_visuals)
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()