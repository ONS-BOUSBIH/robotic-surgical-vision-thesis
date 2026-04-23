from src.Stereo_matching.inference.Stereo_matcher_inferencer import StereoMatcherInferencer
import cv2
import numpy as np

class SGBMInferencer(StereoMatcherInferencer):
    def __init__(self, num_disparities=192, block_size=7, device='cpu'): #104
        super().__init__(device)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities, 
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2, #32
            mode=cv2.StereoSGBM_MODE_HH
        )

    def get_disparity(self, rect_l, rect_r):
        # SGBM works on grayscale or BGR
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        
        # SGBM returns 16-bit fixed point (multiplied by 16)
        disp = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disp