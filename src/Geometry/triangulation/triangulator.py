import os
import cv2
import glob
import configparser
import numpy as np
import zipfile
from src.Keypoints_detection.inference.inferencer import run_multi_tool_inference
from src.Geometry.triangulation.triangulation_utils import get_first_digit

class Triangulator:
    def __init__(self, num_keypoints=7):
        self.num_kpts = num_keypoints
        # Calibration storage
        self.P_l, self.P_r = None, None
        self.K_l, self.K_r = None, None
        self.D_l, self.D_r = None, None

    def load_calibration(self, zip_path):
        """Extracts and parses stereo calibration from the .ini file inside a ZIP."""
        with zipfile.ZipFile(zip_path, "r") as z:
            ini_files = [f for f in z.namelist() if f.endswith(".ini")]
            if not ini_files:
                raise ValueError(f"No INI file found in {zip_path}")
            
            with z.open(ini_files[0]) as ini_file:
                config_bytes = ini_file.read().decode("utf-8")
        
        config = configparser.ConfigParser()
        config.read_string(config_bytes)

        def get_cam_params(section):
            K = np.array([[float(config[section]["fc_x"]), 0, float(config[section]["cc_x"])],
                          [0, float(config[section]["fc_y"]), float(config[section]["cc_y"])],
                          [0, 0, 1]])
            R = np.array([float(config[section][f"R_{i}"]) for i in range(9)]).reshape(3, 3)
            T = np.array([float(config[section][f"T_{i}"]) for i in range(3)]).reshape(3, 1)
            D = np.array([float(config[section][f"kc_{i}"]) for i in range(5)])
            P = K @ np.hstack([R, T])
            return P, K, D

        self.P_l, self.K_l, self.D_l = get_cam_params("StereoLeft")
        self.P_r, self.K_r, self.D_r = get_cam_params("StereoRight")
        return self.P_l, self.P_r

    def undistort_points(self, kpts, side='left'):
        """Undistorts (N, K, 2) or (K, 2) keypoints."""
        K = self.K_l if side == 'left' else self.K_r
        D = self.D_l if side == 'left' else self.D_r
        
        # Flatten if multi-tool (N, Tools, K, 2) -> (N*Tools, K, 2)
        original_shape = kpts.shape
        if len(original_shape) == 4:
            kpts = kpts.reshape(-1, self.num_kpts, 2)

        undistorted = np.zeros_like(kpts)
        for i in range(len(kpts)):
            # Handle (0,0) by not undistorting or marking as NaN
            if np.all(kpts[i] == 0):
                undistorted[i] = np.nan
            else:
                pts_undist = cv2.undistortPoints(kpts[i].astype(np.float32), K, D, P=K)
                undistorted[i] = pts_undist.squeeze(1)
        
        return undistorted.reshape(original_shape)

    def triangulate(self, pts_l, pts_r, masks_l, masks_r):
        """
        pts_l, pts_r: (N, K, 2) undistorted coordinates
        masks_l, masks_r: (N, K) binary visibility masks
        """
        N, K, _ = pts_l.shape
        pts_3d = np.full((N, K, 3), np.nan)

        for i in range(N):
            # Only triangulate points visible in BOTH cameras
            valid_mask = (masks_l[i] > 0) & (masks_r[i] > 0)
            if np.any(valid_mask):
                pl_valid = pts_l[i][valid_mask].T
                pr_valid = pts_r[i][valid_mask].T
                X_h = cv2.triangulatePoints(self.P_l, self.P_r, pl_valid, pr_valid)
                pts_3d[i][valid_mask] = (X_h[:3] / X_h[3]).T
        
        return pts_3d

    def get_reprojection_error(self, X_3D, X_2D_l, X_2D_r):
        """Computes pixel error between reprojected 3D points and detected 2D points."""
        N, K, _ = X_3D.shape
        err_l, err_r = np.full((N, K), np.nan), np.full((N, K), np.nan)
        
        for i in range(N):
            # Convert to homogeneous
            valid = ~np.isnan(X_3D[i, :, 0])
            if not np.any(valid): continue
            
            X_h = np.hstack([X_3D[i, valid], np.ones((np.sum(valid), 1))]).T

            # Project to Left
            proj_l_h = self.P_l @ X_h
            proj_l = (proj_l_h[:2] / proj_l_h[2]).T
            err_l[i, valid] = np.linalg.norm(proj_l - X_2D_l[i, valid], axis=1)

            # Project to Right
            proj_r_h = self.P_r @ X_h
            proj_r = (proj_r_h[:2] / proj_r_h[2]).T
            err_r[i, valid] = np.linalg.norm(proj_r - X_2D_r[i, valid], axis=1)

        return err_l, err_r

    def project_points(self, X_3D, side='left'):
        """
        Projects 3D points back to 2D image plane.
        X_3D: array of shape (N, K, 3) or (K, 3)
        """
        # Handle single tool (K, 3) by adding a temporary batch dimension
        is_single_tool = (X_3D.ndim == 2)
        if is_single_tool:
            X_3D = X_3D[np.newaxis, ...] # Change (K, 3) to (1, K, 3)

        N, K, _ = X_3D.shape
        # Initialize the output array with NaNs
        projected_2d = np.full((N, K, 2), np.nan)

        # Select the correct projection matrix once outside the loop
        P = self.P_l if side.lower() == 'left' else self.P_r

        for i in range(N):
            # Convert to homogeneous
            valid = ~np.isnan(X_3D[i, :, 0])
            if not np.any(valid): 
                continue
            
            # Build homogeneous coordinates (x, y, z, 1)
            X_h = np.hstack([X_3D[i, valid], np.ones((np.sum(valid), 1))]).T

            # Project: (3, 4) @ (4, V) -> (3, V)
            proj_h = P @ X_h
            
            # Homogeneous divide: (x/w, y/w)
            # Resulting 'proj' shape is (V, 2)
            proj = (proj_h[:2] / proj_h[2]).T
            
            # Assign back to the correct indices in our NaN array
            projected_2d[i, valid] = proj

        # Return (K, 2) if input was (K, 3), otherwise (N, K, 2)
        return projected_2d.squeeze(0) if is_single_tool else projected_2d

def run_triangulation_pipeline(
    inferencer, 
    triangulator, 
    test_paths_l, 
    test_paths_r, 
    test_video_list, 
    org_dataset_path, 
    max_tools=2
):
    
    # Run Inference ONLY on test paths
    print("Running batch inference on Test frames...")
    all_preds_l, all_masks_l = run_multi_tool_inference(inferencer, test_paths_l, max_tools)
    all_preds_r, all_masks_r = run_multi_tool_inference(inferencer, test_paths_r, max_tools)

    # Calculate cumulative indices 
    n_frames = {vid: 0 for vid in test_video_list}
    for p in test_paths_r:
        digit = get_first_digit(os.path.basename(p))
        if digit in n_frames: 
            n_frames[digit] += 1

    frame_counts = [n_frames[vid] for vid in test_video_list]
    cumulative = [0] + np.cumsum(frame_counts).tolist()
    
    zip_files = sorted(glob.glob(f"{org_dataset_path}/*.zip"))
    
    # Results Containers
    results = {
        'tri_3d': [[] for _ in range(max_tools)],
        'reproj_err_l': [[] for _ in range(max_tools)],
        'reproj_err_r': [[] for _ in range(max_tools)],
        'preds_l': all_preds_l,
        'preds_r': all_preds_r,
        'video_metadata': []
    }

    # Triangulation Loop 
    for i, video_id in enumerate(test_video_list):
        print(f"Processing Video: {video_id}")
        start, end = cumulative[i], cumulative[i+1]
        
        zip_path = [f for f in zip_files if video_id in os.path.basename(f)][0]
        triangulator.load_calibration(zip_path)

        for t in range(max_tools):
            p_l = all_preds_l[start:end, t]
            p_r = all_preds_r[start:end, t]
            m_l = all_masks_l[start:end, t]
            m_r = all_masks_r[start:end, t]

            undist_l = triangulator.undistort_points(p_l, side='left')
            undist_r = triangulator.undistort_points(p_r, side='right')
            
            pts_3d = triangulator.triangulate(undist_l, undist_r, m_l, m_r)
            err_l, err_r = triangulator.get_reprojection_error(pts_3d, p_l, p_r)

            results['tri_3d'][t].append(pts_3d)
            results['reproj_err_l'][t].append(err_l)
            results['reproj_err_r'][t].append(err_r)
            
        results['video_metadata'].append({'id': video_id, 'range': (start, end)})

    return results

