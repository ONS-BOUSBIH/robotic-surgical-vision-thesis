import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
import math
import os
from sklearn.metrics import jaccard_score
from skimage.metrics import hausdorff_distance
import cv2

def select_random_files(base_results_path, num_samples=10, seed=50, file_path_structure="**/compressed_data/*.npz"):
    """
    Randomly selects a chosen number of file paths following a given structure from a root path.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    base_path = Path(base_results_path)
    
    all_files = sorted(list(base_path.glob(file_path_structure)))
    
    if not all_files:
        print(f"No .npz files found in {base_results_path}")
        return

    selected_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    
    return selected_files

def get_compressed_paths_list(base_results_path, file_path_structure="**/compressed_data/*.npz"):
    """
    Returns the list of the files paths.
    """

    base_path = Path(base_results_path)
    
    all_files = sorted(list(base_path.glob(file_path_structure)))
    
    if not all_files:
        print(f"No .npz files found in {base_results_path}")
        return
    
    return all_files

def plot_selected_files(selected_files, disparities ,title="Examples of Disparity Maps",cols=5, save_path= None, save= False, vid_id_position=0):
    
    num_files = len(selected_files)
    if num_files == 0:
        print("No files selected.")
        return

    # Calculate rows needed based on the number of files and desired columns
    rows = math.ceil(num_files / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.suptitle(title, fontsize=16)
    
    if num_files == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, file_path in enumerate(selected_files):
        video_id = file_path.parents[vid_id_position].name 
        file_stem = file_path.stem 
        frame_number = file_stem.split('_frame_')[-1]
        disp = disparities[i]
        # Plotting
        im = axes_flat[i].imshow(disp, cmap='jet',vmin=0, vmax=190)
        axes_flat[i].set_title(f"Video: {video_id}, frame: {frame_number}", fontsize=10)
        axes_flat[i].axis('off')
        fig.colorbar(im, ax=axes_flat[i], fraction=0.030, pad=0.04)
        
    # Hide any unused subplots 
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    if save:
        assert save_path != None, 'Saving path not given'
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()

def compute_disparities(inferencer, right_img_paths, left_img_root, zip_root, img_shape):
    disparities=[]
    for r_img_path in right_img_paths:
        video_id = r_img_path.parents[0].name 
        l_img_path = os.path.join(left_img_root,video_id,os.path.basename(r_img_path).replace('right','left'))
        zip_path= os.path.join(zip_root,f'{video_id}.zip')
        disp = inferencer.run_inference(l_img_path, r_img_path, zip_path, output_dir=None, img_shape=img_shape, save_png=False)
        disparities.append(disp)
    return disparities

def get_file_paths_from_compressed_file_path(npz_file_path,vid_id_position):

    """
    This function extracts the relevant paths for 3D reconstruction from the name of one selected 
    result compressed file, it follows the local file names.

    """

    video_id = npz_file_path.parents[vid_id_position].name 
    file_stem = npz_file_path.stem 
    frame_number = file_stem.split('_frame_')[-1]
    left_mask_path = f"data/Surgpose_for_segmentation/left_test_set/binary_masks/vid_{video_id}_left_frame_{frame_number}.png"
    right_mask_path = left_mask_path.replace('left', 'right')
    zip_calib_path = f"data/SurgPose/SurgPose_for_HRNet/{video_id}.zip"
    left_img_path = f"data/SurgPose/SurgPose_for_HRNet/Extracted/extracted_frames/{video_id}/vid_{video_id}_left_frame_{frame_number}.jpg"
    right_img_path = f"data/SurgPose/SurgPose_for_HRNet/Extracted_right_test/extracted_frames/{video_id}/vid_{video_id}_right_frame_{frame_number}.jpg"
    frame_id= file_stem.split('.')[0]
    return left_mask_path, right_mask_path, zip_calib_path, left_img_path, right_img_path, frame_id

def compute_hausdorf_distance(m1,m2):
    hd = hausdorff_distance(m1, m2)
    return hd

def compute_dice_score(m1,m2):
    m1 = (m1 > 0).ravel()
    m2 = (m2 > 0).ravel()

    jaccard = jaccard_score(m1, m2, pos_label=1)
    dice_score = (2 * jaccard) / (1 + jaccard)

    return dice_score

def from_disparity_to_mask(disparity_map):
    # Normalize to 8-bit
    disp_min, disp_max = disparity_map.min(), disparity_map.max()
    disp_8bit = ((disparity_map - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
    # Use THRESH_TRIANGLE 
    thresh_val, binary_mask = cv2.threshold(
        disp_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    real_threshold = disp_min + (thresh_val / 255) * (disp_max - disp_min)
    return real_threshold, binary_mask

