# run inference on corresponding left and right frames
# resize the keypoints to the initial frame size
# reconstruct predicted kpts in 3D
from inference import inference
from training import *
import os
import cv2
import glob
import re
import configparser
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_first_digit(string):
    match = re.search(r"\d+", string)
    return match.group() if match else None


def undistort_keypoints(kpts, K, D):
    """ N: number of frames in video
    kpts: (N, 14, 2)
    returns: (N, 14, 2)
    """
    N = kpts.shape[0]
    undist = np.zeros((N, 14, 2))
    for i in range(N):
        kpts_ = kpts[i].astype(np.float32)
        undist[i] = cv2.undistortPoints(kpts_, K, D, P=K).squeeze(1)  # P=K, get undistorted points in pixel coordinates
    return undist

def load_stereo_calibration_from_zip(zip_path):
    """Extract the .ini inside the ZIP and load calibration."""
    with zipfile.ZipFile(zip_path, "r") as z:
        ini_files = [f for f in z.namelist() if f.endswith(".ini")]
        if len(ini_files) == 0:
            raise ValueError(f"No INI file found in {zip_path}")

        ini_name = ini_files[0]
        # extract in memory
        with z.open(ini_name) as ini_file:
            config_bytes = ini_file.read().decode("utf-8")

    # parse ini from string
    config = configparser.ConfigParser()
    config.read_string(config_bytes)

    # Left camera
        ##Intinsic parameters
    fc_x_L = float(config["StereoLeft"]["fc_x"])
    fc_y_L = float(config["StereoLeft"]["fc_y"])
    cc_x_L = float(config["StereoLeft"]["cc_x"])
    cc_y_L = float(config["StereoLeft"]["cc_y"])
        ##Camera matrix
    K_left = np.array([[fc_x_L, 0, cc_x_L],
                       [0, fc_y_L, cc_y_L],
                       [0, 0, 1]])
        ##Extrinsic parameters
    R_left = np.array([float(config["StereoLeft"][f"R_{i}"]) for i in range(9)]).reshape(3, 3)
    T_left = np.array([float(config["StereoLeft"][f"T_{i}"]) for i in range(3)]).reshape(3, 1)
        ##Projection matrix
    P_left = K_left @ np.hstack([R_left, T_left])
        ##Distortion parameters
    D_left = np.array([float(config["StereoLeft"][f"kc_{i}"]) for i in range(5)])

    # Right camera
        ##Intinsic parameters
    fc_x_R = float(config["StereoRight"]["fc_x"])
    fc_y_R = float(config["StereoRight"]["fc_y"])
    cc_x_R = float(config["StereoRight"]["cc_x"])
    cc_y_R = float(config["StereoRight"]["cc_y"])
        ##Camera matrix
    K_right = np.array([[fc_x_R, 0, cc_x_R],
                        [0, fc_y_R, cc_y_R],
                        [0, 0, 1]])
        ##Extrinsic parameters
    R_right = np.array([float(config["StereoRight"][f"R_{i}"]) for i in range(9)]).reshape(3, 3)
    T_right = np.array([float(config["StereoRight"][f"T_{i}"]) for i in range(3)]).reshape(3, 1)
        ##Projection matrix
    P_right = K_right @ np.hstack([R_right, T_right])
        ##Distortion parameters
    D_right = np.array([float(config["StereoRight"][f"kc_{i}"]) for i in range(5)])
  
    return P_left, P_right, K_left, K_right, D_left, D_right
    
def triangulate_points(P_left, P_right, pts_left, pts_right, masks_left, masks_right):
    """
    N: number of frames in video
    pts_left, pts_right: (N, K, 2)
    masks_left, masks_right: (N, K) binary masks
    Returns: (N, K, 3)
    """
    N, K, _ = pts_left.shape
    pts_3d = np.full((N, K, 3), np.nan)  #fill with NaN values

    for i in range(N):
        valid_mask = (masks_left[i] > 0) & (masks_right[i] > 0) #Check if there are any visible keypoints
        if np.any(valid_mask):
            pts_L_valid = pts_left[i][valid_mask].T  #Keep only corrdinates of visible points
            pts_R_valid = pts_right[i][valid_mask].T
            X_h = cv2.triangulatePoints(P_left, P_right, pts_L_valid, pts_R_valid)
            X_3d = (X_h[:3] / X_h[3]).T
            pts_3d[i][valid_mask] = X_3d
    return pts_3d


def reproject(X_3D, X_2D_left,X_2D_right,P_left,P_right):
    '''Reproject  3D points for ONE video and compute left and right errors in respect to the original 2D keypoints.
    (N_frames: number of sampled frames from current video).
    X_3D: (N_frames, 14, 3)
    X_2D_left, X_2D_right: (N_frames, 14, 2)
    P_left, P_right : (3, 4)
    '''
    xR=[]
    xL=[]
    err_l = []
    err_r = []
    for i in range(X_3D.shape[0]):
      
        X_h = np.hstack([X_3D[i], np.ones((X_3D.shape[1], 1))]).T 

        # Reproject to left camera
        xL_h = P_left @ X_h 
        xl = (xL_h[:2] / xL_h[2]).T 

        # Reproject to right camera
        xR_h = P_right @ X_h  
        xr = (xR_h[:2] / xR_h[2]).T 

        # Compute pixel reprojection error
        eL = np.linalg.norm(xl - X_2D_left[i], axis=1)
        eR = np.linalg.norm(xr - X_2D_right[i], axis=1)
        xR.append(xr)
        xL.append(xl)
        err_l.append(eL)
        err_r.append(eR)
    return np.array(err_l), np.array(err_r), np.array(xL), np.array(xR)    

def resize_preds (preds, org_shape, heatmap_shape):
    ''' Resize keypoints coordinates from heatmap size to original image size'''
    H, W = org_shape
    h, w = heatmap_shape
    resized_preds= preds.copy()
    resized_preds[...,0]= resized_preds[...,0]*(W/w)
    resized_preds[...,1]= resized_preds[...,1]*(H/h)
    # resized_preds[...,1]= resized_preds[...,0]*(W/w)
    # resized_preds[...,0]= resized_preds[...,1]*(H/h)
    return resized_preds

#### Plotting#####
def visualize_keypoints_left_right(
    image_path_left, image_path_right,
    preds_left, reprojected_left,
    preds_right, reprojected_right, 
    targets_left= None, targets_right= None,
    title="Left vs Right Frame", save_path= None, frame_name= None):
    """
    Visualize predicted, reprojected, and target keypoints (if given)
    for left and right stereo frames.
    """

    #load images
    img_L = cv2.imread(image_path_left)
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

    img_R = cv2.imread(image_path_right)
    img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_L)
    axes[1].imshow(img_R)

    axes[0].set_title(title + " (Left)")
    axes[1].set_title(title + " (Right)")

    for axis, preds, reproj, targets in [
        (axes[0], preds_left, reprojected_left, targets_left),
        (axes[1], preds_right, reprojected_right, targets_right)
    ]:
        #test variable to only print the label once
        pred_label_added = False
        reproj_label_added = False
        target_label_added = False

        for i in range(preds.shape[0]):
            px, py = preds[i]
            rx, ry = reproj[i]

            #predicted and reprojected
            if not np.isnan(px) and not np.isnan(py) and  not np.isnan(rx) and not np.isnan(ry):
                axis.scatter( px, py, c='cyan', s=40, label="pred" if not pred_label_added else None)
                axis.text(px+5, py +5, str(i), color='cyan', fontsize=10)
                pred_label_added = True
            
                axis.scatter(rx, ry, c='red', s=40, marker='x', label="reprojected" if not reproj_label_added else None)
                axis.text(rx+ 5, ry + 5, str(i), color='red', fontsize=10)
                reproj_label_added = True

            #targets if given
            if targets is not None:
                tx, ty = targets[i]
                if not np.isnan(tx) and not np.isnan(ty):
                    axis.scatter( tx, ty, c='green', s=40, marker='*',label="target" if not target_label_added else None)
                    axis.text(tx+ 5, ty + 5, str(i), color='green', fontsize=10)
                    target_label_added = True

        axis.legend()
        plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path,f'{frame_name}_reprojection.png'))
        plt.close()
    else:
        plt.show()

def plot_frame_3d(pred_kpts_3d,target_kpts_3d, title="3D Keypoints", edges= False): ###ADD VISIBILITY MASKS
    '''Plots the 3D projected keypoints'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = pred_kpts_3d[:,0]
    ys = pred_kpts_3d[:,1]
    zs = pred_kpts_3d[:,2]

    xt = target_kpts_3d[:,0]
    yt = target_kpts_3d[:,1]
    zt = target_kpts_3d[:,2]
    #Plot edges
    
    

    

    ax.scatter(xs, ys, zs,color= 'blue', s=40, label='prediction')
    ax.scatter(xt, yt, zt,color='red', s=40, label='target')
    if edges:
        edges=[(0,1),(1,2),(2,3),(2,4),(7,8),(8,9),(9,10),(9,11)]
        for i,j in edges:
            ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            [zs[i], zs[j]],
            color='blue')

            ax.plot(
            [xt[i], xt[j]],
            [yt[i], yt[j]],
            [zt[i], zt[j]],
            color='red')
    for i, (x,y,z) in enumerate(pred_kpts_3d):
        ax.text(x, y, z, str(i), color='blue')
    
    for i, (x,y,z) in enumerate(target_kpts_3d):
        ax.text(x, y, z, str(i), color='red')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.view_init(elev=30, azim=-80)
    plt.legend()
    plt.show()

def main(ckpt_path, input_shape, heatmap_shape):
    cfg_file = '/srv/homes/onbo10/thesis_Ons/HRNet-experiments/HRNet_finetuned/w32_256x192_adam_lr1e-3_out14-finetune.yaml' #'/srv/homes/onbo10/thesis_Ons/HRNet-experiments/HRNet_finetuned/w48_384x288_adam_lr1e-3_out14-finetune.yaml'
    model_weights=ckpt_path #"./Experiment2/training_chekpoints2025-11-12_13-57-19/model_epoch50.pth"
    #"./Experiment1/training_chekpoints2025-11-07_12-30-59/model_epoch30.pth"
    data_root_left='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted' #'/srv/homes/onbo10/thesis_Ons/MiniSurgPose/Extracted3'  #
    data_root_right ='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_right_test' #'/srv/homes/onbo10/thesis_Ons/MiniSurgPose/Extracted3_right' #
    split_file= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted/video_split.yaml' #'/srv/homes/onbo10/thesis_Ons/MiniSurgPose/Extracted3/video_split.yaml' #
    org_dataset_path = '/srv/homes/onbo10/thesis_Ons/SurgePoseData'
    #heatmap_shape = (64, 48)
    threshold= 0.2
    
    #Load finetuned model and get device
    model= load_pretrained_HRNet(cfg_file,model_weights, finetuned=True)
    device= get_device()
    
    #Inference on left frames
    all_preds_left, all_targets_left, all_gt_masks_left, all_maxvals_left= inference(model,device,data_root_left, split_file, input_shape,heatmap_shape,custom_arg_max= True, threshold=0.2)
    
    #Inference on right frames
    all_preds_right, all_targets_right, all_gt_masks_right, all_maxvals_right= inference(model,device,data_root_right, split_file, input_shape,heatmap_shape,custom_arg_max= True, threshold=0.2)
    
    #Get the original shape of the images
    sample_img_path = sorted(glob.glob(f'{data_root_right}/**/*.jpg',recursive=True))[0]
    sample_img = cv2.imread(sample_img_path)
    org_shape = sample_img.shape[:2]
    print('org:', org_shape)
    print('heatmap:',heatmap_shape)
    #Resize left and right predicted keypoints 
   
    all_preds_left= resize_preds(all_preds_left,org_shape, heatmap_shape) 
    all_preds_right= resize_preds(all_preds_right,org_shape, heatmap_shape)
    # Get prediction masks (valid argmax values)
    all_mask_pred_left = (all_maxvals_left > threshold).astype(np.float64)
    
    all_mask_pred_right= (all_maxvals_right> threshold).astype(np.float64)
   
    all_targets_left= resize_preds(all_targets_left,org_shape, heatmap_shape) 
    all_targets_right= resize_preds(all_targets_right,org_shape, heatmap_shape)
    
    img_paths = sorted(glob.glob(f'{data_root_right}/**/*.jpg',recursive=True))
    img_paths= [os.path.basename(p) for p in img_paths]
    
    with open(split_file, "r") as f:
        splits = yaml.safe_load(f)
    test_video_list = splits['test']    
    
    # Count frames per video ID
    video_ids = test_video_list
    n_frames = {vid: 0 for vid in video_ids}

    for p in img_paths:
        digit = get_first_digit(p)
        if digit in n_frames:
            n_frames[digit] += 1

    # Turn dictionary into ordered list aligned with test_video_list
    frame_counts = [n_frames[vid] for vid in test_video_list]

    # Build cumulative index for slicing predictions
    cumulative = [0]
    for c in frame_counts:
        cumulative.append(cumulative[-1] + c)

    zip_files = sorted(glob.glob(f"{org_dataset_path}/*.zip"))

    triangulated_all_pred = []
    triangulated_all_target = []
    P_left_matrices = []
    P_right_matrices = []
    undis_targets_L =[]
    undis_targets_R =[]
    undis_preds_L =[]
    undis_preds_R =[]
    targets_L =[]
    targets_R =[]
    preds_L =[]
    preds_R =[]

    for i, video_id in enumerate(test_video_list):

        start, end = cumulative[i], cumulative[i+1]
        pred_L = all_preds_left[start:end]
        pred_R = all_preds_right[start:end]

        target_L = all_targets_left[start:end]
        target_R = all_targets_right[start:end]

      
        mask_L= all_mask_pred_left[start:end]
        mask_R = all_mask_pred_right[start:end]

        gt_mask_L= all_gt_masks_left[start:end]
        gt_mask_R = all_gt_masks_right[start:end]
        # Match ZIP file for this video ID
        zip_path = [f for f in zip_files if video_id in os.path.basename(f)][0]

        # Load calibration
        P_l, P_r, K_l, K_r, D_l, D_r = load_stereo_calibration_from_zip(zip_path)

        #Undistort keypoints

        undis_pred_L = undistort_keypoints(pred_L, K_l, D_l)
        undis_pred_R = undistort_keypoints(pred_R, K_r, D_r)

        undis_target_L = undistort_keypoints(target_L, K_l, D_l)
        undis_target_R = undistort_keypoints(target_R, K_r, D_r)

        # Triangulate
        tri_3d_pred = triangulate_points(P_l, P_r, undis_pred_L, undis_pred_R,mask_L, mask_R)
        tri_3d_target = triangulate_points(P_l, P_r, undis_target_L, undis_target_R, gt_mask_L, gt_mask_R)
        #tri_3d_target = triangulate_points(P_l, P_r, target_L, target_R, mask_L, mask_R)

        triangulated_all_pred.append(tri_3d_pred)
        triangulated_all_target.append(tri_3d_target)
        P_left_matrices.append(P_l)
        P_right_matrices.append(P_r)
        
        undis_targets_L.append(undis_target_L)
        undis_targets_R.append(undis_target_R)
        undis_preds_L.append(undis_pred_L)
        undis_preds_R.append(undis_pred_R)
        targets_L.append(target_L)
        targets_R.append(target_R)
        preds_L.append(pred_L)
        preds_R.append(pred_R)


    print("Triangulation complete.")
    return triangulated_all_pred , triangulated_all_target , P_left_matrices, P_right_matrices ,undis_preds_L, undis_targets_L, undis_preds_R, undis_targets_R , preds_L, targets_L, preds_R, targets_R




    



