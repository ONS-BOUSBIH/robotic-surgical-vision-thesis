import os
import cv2
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
            # Original: vid_000033_frame_000000.jpg
            # nnU-Net:  sp_000033_frame_000000_0000.png
            
            # Remove extension
            base_name = os.path.splitext(frame_name)[0]
            # Replace 'vid' with 'sp' 
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

def convert_surgpose_to_nnunet(source_root,source_root_bboxes, target_root, yaml_target_root, test_video_ids, max_per_vid=None, seed=50):
    """
    source_root: Path containing video ID folders (000033, etc.)
    target_root: The 'imagesTs' folder for nnU-Net images
    yaml_target_root: Folder to store stripped YAML bounding box files
    """
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(yaml_target_root, exist_ok=True)
    random.seed(seed)

    for vid_id in test_video_ids:
        vid_folder_path = os.path.join(source_root, vid_id)
        vid_folder_path_bboxes=  os.path.join(source_root_bboxes, vid_id)
        if not os.path.isdir(vid_folder_path):
            continue

        # Get all jpg frames
        frames = [f for f in os.listdir(vid_folder_path) if f.lower().endswith('.jpg')]
        
        if max_per_vid and len(frames) > max_per_vid:
            selected_frames = random.sample(frames, max_per_vid)
        else:
            selected_frames = frames

        for frame_name in tqdm(selected_frames, desc=f"Processing Video {vid_id}"):
            base_name = os.path.splitext(frame_name)[0] # e.g., vid_000033_frame_000000
            clean_name = base_name.replace('vid_', 'sp_')
            
            # --- Handle Image ---
            nnunet_img_name = f"{clean_name}_0000.png"
            source_img_path = os.path.join(vid_folder_path, frame_name)
            target_img_path = os.path.join(target_root, nnunet_img_name)

            img = cv2.imread(source_img_path)
            if img is not None:
                cv2.imwrite(target_img_path, img)

            # --- Handle YAML (Bounding Boxes) ---
            source_yaml_path = os.path.join(vid_folder_path_bboxes, f"{base_name}.yaml")
            target_yaml_path = os.path.join(yaml_target_root, f"{clean_name}.yaml")

            if os.path.exists(source_yaml_path):
                with open(source_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Strip keypoints and visibility, keep only ID and BBox
                if 'objects' in data:
                    new_objects = []
                    for obj in data['objects']:
                        new_obj = {
                            'id': obj.get('id'),
                            'bbox': obj.get('bbox')
                        }
                        new_objects.append(new_obj)
                    data['objects'] = new_objects

                # Save the stripped version
                with open(target_yaml_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    source_dir_frame = "data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/extracted_frames"
    source_dir_bbox = "data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/extracted_bboxes_kpts"
    target_dir = "data/EndoVis2017/Surgpose_for_nnUnet_test_left_right/imagesTs"
    yaml_target_dir = "data/EndoVis2017/Surgpose_for_nnUnet_test_left_right/bboxesTs"
    
    split_path = 'data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/video_split.yaml'
    with open(split_path, "r") as f:
        splits = yaml.safe_load(f)
    
    test_video_list = splits['test']
    convert_surgpose_to_nnunet(source_dir_frame,source_dir_bbox, target_dir, yaml_target_dir, test_video_list, max_per_vid=30)


# if __name__ == "__main__":
#     source_dir = "data/SurgPose/SurgPose_for_HRNet/Extracted_right_test/extracted_frames"
#     target_dir = "data/EndoVis2017/Surgpose_for_nnUnet_test/imagesTs"
#     split_path='data/SurgPose/SurgPose_for_HRNet/Extracted_left_right/video_split.yaml'
#     with open(split_path, "r") as f:
#         splits = yaml.safe_load(f)
#     test_video_list = splits['test']
#     convert_surgpose_to_nnunet(source_dir, target_dir, test_video_list, max_per_vid=30)