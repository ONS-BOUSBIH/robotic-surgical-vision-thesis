import os
import cv2
from pathlib import Path
from tqdm import tqdm
import yaml
import random 

def convert_surgpose_to_nnunet(source_root, target_root, test_video_ids,max_per_vid=None, seed=50):
    """
    source_root: Path containing the video ID folders (000033, etc.)
    target_root: The 'imagesTs' folder for your nnU-Net inference
    test_video_ids: List of folder names to process
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_root, exist_ok=True)
    random.seed(seed)
    print(f"Starting conversion for {len(test_video_ids)} videos...")

    for vid_id in test_video_ids:
        vid_folder_path = os.path.join(source_root, vid_id)
        
        if not os.path.isdir(vid_folder_path):
            print(f"Skipping {vid_id}: Folder not found.")
            continue

        # Get all jpg frames in the folder
        frames = [f for f in os.listdir(vid_folder_path) if f.lower().endswith('.jpg')]
        if max_per_vid:
            frames.sort()

            # Perform the sampling
            if len(frames) > max_per_vid:
                selected_frames = random.sample(frames, max_per_vid)
            else:
                selected_frames = frames
                print(f"Note: Video {vid_id} only has {len(frames)} frames. Taking all.")
        else:
            selected_frames= frames
        for frame_name in tqdm(selected_frames, desc=f"Processing Video {vid_id}"):
            # 1. Logic for New Naming Convention
            # Original: vid_000033_frame_000000.jpg
            # nnU-Net:  sp_000033_frame_000000_0000.png
            
            # Remove extension
            base_name = os.path.splitext(frame_name)[0]
            # Replace 'vid' with 'sp' (short for SurgPose) to avoid confusion with EndoVis
            clean_name = base_name.replace('vid_', 'sp_')
            # Add nnU-Net channel suffix and new extension
            nnunet_name = f"{clean_name}_0000.png"
            
            source_file = os.path.join(vid_folder_path, frame_name)
            target_file = os.path.join(target_root, nnunet_name)

       
            img = cv2.imread(source_file)
            if img is not None:
                cv2.imwrite(target_file, img)
            else:
                print(f"Error reading {source_file}")

    print(f"\nSuccess! All frames are now in: {target_root}")


if __name__ == "__main__":
    source_dir = "data/SurgPose/SurgPose_for_HRNet/Extracted_right_test/extracted_frames"
    target_dir = "data/EndoVis2017/Surgpose_for_nnUnet_test/imagesTs"
    split_path='data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/video_split.yaml'
    with open(split_path, "r") as f:
        splits = yaml.safe_load(f)
    test_video_list = splits['test']
    convert_surgpose_to_nnunet(source_dir, target_dir, test_video_list, max_per_vid=30)