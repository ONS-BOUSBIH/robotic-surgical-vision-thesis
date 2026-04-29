from src.Stereo_matching.inference.Stereo_matcher_inferencer import StereoMatcherInferencer

import torch
import numpy as np
import cv2
import os
from s2m2.core.utils.model_utils import load_model, run_stereo_matching

class S2M2Inferencer(StereoMatcherInferencer):
    def __init__(self, checkpoint_path, model_type='S', num_refine=3, allow_negative=False, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device_obj = torch.device('cpu')
        else:
            self.device_obj = torch.device(device)
        super().__init__(str(self.device_obj))
        # as the load_model function expects a directory path
        self.model = load_model(
            checkpoint_path, 
            model_type, 
            not allow_negative, 
            num_refine, 
            self.device_obj
        )
        self.model.eval()

    def get_disparity(self, rect_l, rect_r):
        """
        Adapts the rectification images to the S2M2 utility expectations.
        """
        # Prepare dimensions 
        h, w = rect_l.shape[:2]
        img_height = (h // 32) * 32
        img_width = (w // 32) * 32
        
        # Crop to the multiple of 32 
        left_cropped = rect_l[:img_height, :img_width]
        right_cropped = rect_r[:img_height, :img_width]

        # Convert to torch tensors [1, 3, H, W]
        left_torch = torch.from_numpy(left_cropped).permute(2, 0, 1).unsqueeze(0).to(self.device)
        right_torch = torch.from_numpy(right_cropped).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run inference 
        # This function handles the forward pass and returns a tuple
        with torch.no_grad():
            _ = run_stereo_matching(self.model, left_torch, right_torch, self.device_obj) #pre-run
            pred_disp, pred_occ, pred_conf, avg_conf, avg_time = run_stereo_matching(
                self.model, left_torch, right_torch, self.device_obj, N_repeat=5
            )

        # Convert back to numpy and resize to original input size if cropped
        disparity = pred_disp.squeeze().cpu().numpy()
        
        if img_height != h or img_width != w:
            # Pad with zeros to return to original resolution so LRC check works
            full_disp = np.zeros((h, w), dtype=np.float32)
            full_disp[:img_height, :img_width] = disparity
            return full_disp

        return disparity