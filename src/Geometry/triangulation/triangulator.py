import cv2
import configparser
import numpy as np
import zipfile


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
            return P, K, D, R, T

        
        self.P_l, self.K_l, self.D_l, R_l, T_l = get_cam_params("StereoLeft")
        self.P_r, self.K_r, self.D_r, R_r, T_r = get_cam_params("StereoRight")
        self.R=  R_r
        self.T= T_r
        # Compute Essential Matrix using RELATIVE params
        t = T_r.flatten()
        t_skew = np.array([[0, -t[2], t[1]],
                           [t[2], 0, -t[0]],
                           [-t[1], t[0], 0]])
        E = t_skew @ R_r

        # Compute Fundamental Matrix
        self.F = np.linalg.inv(self.K_r).T @ E @ np.linalg.inv(self.K_l)

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
    

    def get_rectification_maps(self, img_size, mode='conventional'):
        """
        Calculates maps for rectification.
        'conventional': Full 3D rectification (Undistort + Rotate).
        'pseudo': 2D translation only (Shift to align centers).
        """
        self.rect_mode = mode
        h, w = img_size

        if mode == 'conventional':
            lkmat = np.array([
                [self.K_l[0, 0], 0, self.K_l[0, 2]],
                [0, self.K_l[1, 1], self.K_l[1, 2]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            rkmat = np.array([
                [self.K_r[0, 0], 0, self.K_r[0, 2]],
                [0, self.K_r[1, 1], self.K_r[1, 2]],
                [0, 0, 1]
            ], dtype=np.float64)

            # Perform Stereo Rectify with alpha=0 (removes black borders)
            r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
                cameraMatrix1=lkmat, 
                distCoeffs1=self.D_l, 
                cameraMatrix2=rkmat, 
                distCoeffs2=self.D_r, 
                imageSize=(w, h), 
                R=self.R, 
                T=self.T.reshape(3, 1),
                alpha=0 
            )

            # Generate Undistort and Rectify Maps
            lmap1, lmap2 = cv2.initUndistortRectifyMap(
                lkmat, self.D_l, r1, p1, (w, h), cv2.CV_32FC1
            )
            rmap1, rmap2 = cv2.initUndistortRectifyMap(
                rkmat, self.D_r, r2, p2, (w, h), cv2.CV_32FC1
            )

            return lmap1, lmap2, rmap1, rmap2, q

        elif mode == 'pseudo':
            # Calculate the 2D shift (dx, dy) to align principal points
            dx = self.K_l[0, 2] - self.K_r[0, 2]
            dy = self.K_l[1, 2] - self.K_r[1, 2]
            
            # Create affine translation matrix
            # Shifting the right image to match the left
            pseudo_matrix = np.array([
                [1, 0, dx], 
                [0, 1, dy]
            ], dtype=np.float32)
            
            # Return dummy values for maps and a basic Q matrix
            return None, None, pseudo_matrix, None, np.eye(4)
    
    def rectify_images(self, img_l, img_r, lmap1, lmap2, r_map1_or_mat, r_map2, rect_mode="conventional"):
        """
        Applies the rectification.
        In 'conventional' mode: r_map1_or_mat is the right image x map.
        In 'pseudo' mode: r_map1_or_mat is the 2D translation matrix.
        """
        if rect_mode == 'conventional':
            rect_l = cv2.remap(img_l, lmap1, lmap2, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, r_map1_or_mat, r_map2, cv2.INTER_LINEAR)
            return rect_l, rect_r

        elif rect_mode == 'pseudo':
            h, w = img_l.shape[:2]
            # Left image stays raw, right image is shifted
            rect_l = img_l.copy()
            rect_r = cv2.warpAffine(img_r, r_map1_or_mat, (w, h))
            return rect_l, rect_r
    
    def reconstruct_3d_sgbm(self, rect_l, rect_r, Q, rect_mask_l):
        """
        Dense reconstruction using SGBM.
        """
        # SGBM requires grayscale
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        window_size = 7
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=192, 
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100, # Removes noise
            speckleRange=2,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY
        )

       
        # Disparity Calculation
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        points_3d_full = cv2.reprojectImageTo3D(disparity, Q)

        # Filter by your masks and valid depth
        valid_mask =  (rect_mask_l > 0) &(disparity > 0) 
        point_cloud = points_3d_full[valid_mask]
        
        # Pull colors from the rectified image for the cloud
        colors = rect_l[valid_mask][:, ::-1]/255.0 

        # Final filtering to remove infinite/extreme values 
        z_filter = (point_cloud[:, 2] > 0) & (point_cloud[:, 2] < 500)
        
        return point_cloud[z_filter], colors[z_filter], disparity

    def reconstruct_3d_sgbm_masked(self, rect_l, rect_r, Q, rect_mask_l, rect_mask_r):
            """
            Dense reconstruction using SGBM and masking the tools first.
            """
            # SGBM requires grayscale
            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
            masked_l = gray_l* rect_mask_l
            masked_r = gray_r* rect_mask_r
            window_size = 7
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=192, 
                blockSize=window_size,
                P1=8 * 3 * window_size**2,
                P2=32 * 3 * window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100, # Removes noise
                speckleRange=2,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY
            )

        
            # Disparity Calculation
            disparity = stereo.compute(masked_l, masked_r).astype(np.float32) / 16.0

            points_3d_full = cv2.reprojectImageTo3D(disparity, Q)

            # Filter by your masks and valid depth
            valid_mask =  (rect_mask_l > 0) &(disparity > 0) 
            point_cloud = points_3d_full[valid_mask]
            
            # Pull colors from the rectified image for the cloud
            colors = rect_l[valid_mask][:, ::-1]/255.0 

            # Final filtering to remove infinite/extreme values 
            z_filter = (point_cloud[:, 2] > 0) & (point_cloud[:, 2] < 500)
            
            return point_cloud[z_filter], colors[z_filter], disparity

    def project_disparity_to_3d(self, disparity, q_matrix, rect_l, rect_mask_l):
        
        disparity[rect_mask_l == 0] = 0
        points_3d = cv2.reprojectImageTo3D(disparity, q_matrix)
        valid_mask =  (rect_mask_l > 0) &(disparity > 0) 
        point_cloud = points_3d[valid_mask]
            
        # Pull colors from the rectified image for the cloud
        colors = rect_l[valid_mask][:, ::-1]/255.0 

        # Final filtering to remove infinite/extreme values 
        z_filter = (point_cloud[:, 2] > 0) & (point_cloud[:, 2] < 500)
        
        return point_cloud[z_filter], colors[z_filter], disparity



