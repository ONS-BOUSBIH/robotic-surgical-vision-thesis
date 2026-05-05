import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
import math


def select_random_files(base_results_path, num_samples=10, seed=50, file_path_structure="**/compressed_data/*.npz"):
    """
    Randomly selects a chosen number of file paths following a given structure from a root path.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    base_path = Path(base_results_path)
    
    all_files = list(base_path.glob(file_path_structure))
    
    if not all_files:
        print(f"No .npz files found in {base_results_path}")
        return

    selected_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    
    return selected_files


def plot_selected_files(selected_files, title="Examples of Disparity Maps",cols=5, save_path= None, save= False):
    
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
        # Load the data
        data = np.load(file_path)
        disp = data['disparity']
        # Plotting
        im = axes_flat[i].imshow(disp, cmap='jet')
        axes_flat[i].set_title(f"File: {file_path.stem}", fontsize=10)
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

def get_file_paths_from_compressed_file_path(npz_file_path):

    """
    This function extracts the relevant paths for 3D reconstruction from the name of one selected 
    result compressed file, it follows the local file names.

    """
    
    video_id = npz_file_path.parents[1].name 
    file_stem = npz_file_path.stem 
    frame_number = file_stem.split('_frame_')[-1]
    left_mask_path = f"data/Surgpose_for_segmentation/left_test_set/binary_masks/vid_{video_id}_left_frame_{frame_number}.png"
    right_mask_path = left_mask_path.replace('left', 'right')
    zip_calib_path = f"data/SurgPose/SurgPose_for_HRNet/{video_id}.zip"
    left_img_path = f"data/SurgPose/SurgPose_for_HRNet/Extracted/extracted_frames/{video_id}/vid_{video_id}_left_frame_{frame_number}.jpg"
    right_img_path = f"data/SurgPose/SurgPose_for_HRNet/Extracted_right_test/extracted_frames/{video_id}/vid_{video_id}_right_frame_{frame_number}.jpg"
    frame_id= file_stem.split('.')[0]
    return left_mask_path, right_mask_path, zip_calib_path, left_img_path, right_img_path, frame_id
