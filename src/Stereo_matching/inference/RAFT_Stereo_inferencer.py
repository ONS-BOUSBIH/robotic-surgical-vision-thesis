import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] 
RAFT_PATH = ROOT / "external" / "RAFT-Stereo"

if str(RAFT_PATH) not in sys.path:
    sys.path.append(str(RAFT_PATH))
    sys.path.append(str(RAFT_PATH / "core"))

from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import torch
from src.Stereo_matching.inference.Stereo_matcher_inferencer import StereoMatcherInferencer

class RAFTSTEREOInferencer(StereoMatcherInferencer):
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__(device)
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
        # BGR to RGB 
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

  
      

   

    

    
  
