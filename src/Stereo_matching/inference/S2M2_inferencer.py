from src.Stereo_matching.inference.Stereo_matcher_inferencer import StereoMatcherInferencer
import torch
import numpy as np
import cv2
import os
from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from torch.functional import F

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
       
        h, w = rect_l.shape[:2]
        # Calculate total padding needed
        pad_h_total = (32 - h % 32) % 32
        pad_w_total = (32 - w % 32) % 32
        
        # Split padding for symmetry
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        # Convert to tensors
        t_l = torch.from_numpy(rect_l).permute(2, 0, 1).unsqueeze(0).to(self.device_obj).float()
        t_r = torch.from_numpy(rect_r).permute(2, 0, 1).unsqueeze(0).to(self.device_obj).float()

        # Apply Symmetric Padding: (left, right, top, bottom)
        t_l_padded = F.pad(t_l, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        t_r_padded = F.pad(t_r, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # Inference
        with torch.no_grad():
            _ = run_stereo_matching(self.model, t_l_padded, t_r_padded, self.device_obj) #pre-run
            pred_disp, pred_occ, pred_conf, avg_conf, avg_time = run_stereo_matching(
                self.model, t_l_padded, t_r_padded, self.device_obj, N_repeat=5
            )

        # Symmetric Crop back to original size
        disparity = pred_disp.squeeze().cpu().numpy()
        
        # Crop using the same indices used for padding
        end_h = disparity.shape[0] - pad_bottom
        end_w = disparity.shape[1] - pad_right
        
        disparity_final = disparity[pad_top:end_h, pad_left:end_w]

        return disparity_final