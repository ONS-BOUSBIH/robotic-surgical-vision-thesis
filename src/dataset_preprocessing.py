import zipfile
import yaml
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random


def extract_frames():
    dataset_path= '/srv/homes/onbo10/thesis_Ons/MiniSurgPose'  # "/srv/homes/onbo10/thesis_Ons/SurgePoseData" 
    output_dir_jpg = dataset_path + "/Extracted/extracted_frames"
    output_dir_kp = dataset_path + "/Extracted/extracted_keypoints"

    T=300 #Number of frames kept per video
    num_keypoints = 14 #Max number of keypoints per frame in this dataset

    if not os.path.exists(output_dir_jpg):
        os.makedirs(output_dir_jpg)
    if not os.path.exists(output_dir_kp):
        os.makedirs(output_dir_kp)

    zip_files = sorted(glob.glob(f'{dataset_path}/*.zip'))

    for zip_path in tqdm(zip_files, desc="Processing videos", unit="video"):   
        
        vid_id = os.path.splitext(os.path.basename(zip_path))[0] #get video ID

        current_jpg = os.path.join(output_dir_jpg,vid_id)
        current_kp = os.path.join(output_dir_kp,vid_id)

        if not os.path.exists(current_jpg):
            os.mkdir(current_jpg)
        if not os.path.exists(current_kp):
            os.mkdir(current_kp)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # video file
            video_file = [f for f in zip_ref.namelist() if (f.endswith(".mp4") and 'left'in f.lower())][0]
            # Yaml annotation file
            yaml_file = [f for f in zip_ref.namelist() if "keypoints_left" in f.lower()][0]
            
            with zip_ref.open(video_file) as f:
                video_bytes = f.read()
            with zip_ref.open(yaml_file) as f:
                keypoints_data = yaml.safe_load(f)
            
            keypoints_data = {int(k): v for k, v in keypoints_data.items()}

            total_frames = len(sorted(keypoints_data.keys())) #total number of frames for current video
            pace = max(1, total_frames // T) # Compute the interval width for selecting frames
            
            tmp_path = f"/tmp/{vid_id}_temp.mp4"
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(video_bytes)
            cap = cv2.VideoCapture(tmp_path)
            # Select idx for frames to keep
            valid_frames = sorted(keypoints_data.keys())[::pace]

            for frame_idx in valid_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                #save only current valid frame
                save_path = os.path.join(current_jpg, f"vid_{vid_id}_frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, frame)
                #get keypoints for current frame
                #frame_dict = keypoints_data.get(frame_idx, {})
                frame_dict = keypoints_data[frame_idx]
                keypoints = []
                visibility = []
                # get the key points and create visibility masks for the present kpts/14 keypoints
                for k in range(1, num_keypoints + 1):
                    if k in frame_dict and frame_dict[k] is not None:
                        keypoints.append(frame_dict[k])
                        visibility.append(1)
                    else:
                        keypoints.append([0, 0])
                        visibility.append(0)
                
                ann_path = os.path.join(current_kp, f"vid_{vid_id}_frame_{frame_idx:06d}.yaml")
                # Save keypoints and visibility masks
                ann_data = {"video_id": vid_id,
                            "frame_id": frame_idx,
                            "keypoints": keypoints,      
                            "visibility": visibility   
                        }

                with open(ann_path, 'w') as f:
                    yaml.safe_dump(ann_data, f)
            cap.release()
            os.remove(tmp_path)


def extract_frames_alternated():
    dataset_path= "/srv/homes/onbo10/thesis_Ons/SurgePoseData" #'/srv/homes/onbo10/thesis_Ons/MiniSurgPose'
    output_dir_jpg = dataset_path + "/Extracted_left_right/extracted_frames"
    output_dir_kp = dataset_path + "/Extracted_left_right/extracted_keypoints"

    T=300 #Number of frames kept per video
    num_keypoints = 14 #Max number of keypoints per frame in this dataset

    if not os.path.exists(output_dir_jpg):
        os.makedirs(output_dir_jpg)
    if not os.path.exists(output_dir_kp):
        os.makedirs(output_dir_kp)

    zip_files = sorted(glob.glob(f'{dataset_path}/*.zip'))

    for zip_path in tqdm(zip_files, desc="Processing videos", unit="video"):   
        
        vid_id = os.path.splitext(os.path.basename(zip_path))[0] #get video ID

        current_jpg = os.path.join(output_dir_jpg,vid_id)
        current_kp = os.path.join(output_dir_kp,vid_id)

        if not os.path.exists(current_jpg):
            os.mkdir(current_jpg)
        if not os.path.exists(current_kp):
            os.mkdir(current_kp)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # video file
            video_file_left = [f for f in zip_ref.namelist() if (f.endswith(".mp4") and 'left'in f.lower())][0]
            # Yaml annotation file
            yaml_file_left = [f for f in zip_ref.namelist() if "keypoints_left" in f.lower()][0]
            # video file
            video_file_right = [f for f in zip_ref.namelist() if (f.endswith(".mp4") and 'right'in f.lower())][0]
            # Yaml annotation file
            yaml_file_right = [f for f in zip_ref.namelist() if "keypoints_right" in f.lower()][0]
            
            with zip_ref.open(video_file_left) as f:
                video_bytes_left = f.read()
            with zip_ref.open(yaml_file_left) as f:
                keypoints_data_left = yaml.safe_load(f)
            
            keypoints_data_left = {int(k): v for k, v in keypoints_data_left.items()}

            with zip_ref.open(video_file_right) as f:
                video_bytes_right = f.read()
            with zip_ref.open(yaml_file_right) as f:
                keypoints_data_right = yaml.safe_load(f)
            
            keypoints_data_right = {int(k): v for k, v in keypoints_data_right.items()}

            all_frames = sorted(set(keypoints_data_left.keys()).union(set(keypoints_data_right.keys())))
            #total_frames = len(sorted(keypoints_data_left.keys())) #total number of frames for current video
            pace = max(1, len(all_frames) // T) # Compute the interval width for selecting frames
            
            tmp_path_left = f"/tmp/{vid_id}_left_temp.mp4"
            with open(tmp_path_left, "wb") as tmp_file_left:
                tmp_file_left.write(video_bytes_left)
            cap_left = cv2.VideoCapture(tmp_path_left)

            tmp_path_right = f"/tmp/{vid_id}_right_temp.mp4"
            with open(tmp_path_right, "wb") as tmp_file_right:
                tmp_file_right.write(video_bytes_right)
            cap_right = cv2.VideoCapture(tmp_path_right)
            # Select idx for frames to keep
            total_valid_frames = all_frames[::pace]
            valid_frames_left = total_valid_frames[::2]
            valid_frames_right = total_valid_frames[1::2]
            for frame_idx in valid_frames_left:
                cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap_left.read()
                if not ret:
                    continue
                
                #save only current valid frame
                save_path = os.path.join(current_jpg, f"vid_{vid_id}_left_frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, frame)
                #get keypoints for current frame
                #frame_dict = keypoints_data.get(frame_idx, {})
                frame_dict = keypoints_data_left[frame_idx]
                keypoints = []
                visibility = []
                # get the key points and create visibility masks for the present kpts/14 keypoints
                for k in range(1, num_keypoints + 1):
                    if k in frame_dict and frame_dict[k] is not None:
                        keypoints.append(frame_dict[k])
                        visibility.append(1)
                    else:
                        keypoints.append([0, 0])
                        visibility.append(0)
                
                ann_path = os.path.join(current_kp, f"vid_{vid_id}_left_frame_{frame_idx:06d}.yaml")
                # Save keypoints and visibility masks
                ann_data = {"video_id": vid_id,
                            "frame_id": frame_idx,
                            "keypoints": keypoints,      
                            "visibility": visibility   
                        }

                with open(ann_path, 'w') as f:
                    yaml.safe_dump(ann_data, f)
            cap_left.release()
            os.remove(tmp_path_left)

            #right
            for frame_idx in valid_frames_right:
                cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap_right.read()
                if not ret:
                    continue
                
                #save only current valid frame
                save_path = os.path.join(current_jpg, f"vid_{vid_id}_right_frame_{frame_idx:06d}.jpg")
                cv2.imwrite(save_path, frame)
                #get keypoints for current frame
                #frame_dict = keypoints_data.get(frame_idx, {})
                frame_dict = keypoints_data_right[frame_idx]
                keypoints = []
                visibility = []
                # get the key points and create visibility masks for the present kpts/14 keypoints
                for k in range(1, num_keypoints + 1):
                    if k in frame_dict and frame_dict[k] is not None:
                        keypoints.append(frame_dict[k])
                        visibility.append(1)
                    else:
                        keypoints.append([0, 0])
                        visibility.append(0)
                
                ann_path = os.path.join(current_kp, f"vid_{vid_id}_right_frame_{frame_idx:06d}.yaml")
                # Save keypoints and visibility masks
                ann_data = {"video_id": vid_id,
                            "frame_id": frame_idx,
                            "keypoints": keypoints,      
                            "visibility": visibility   
                        }

                with open(ann_path, 'w') as f:
                    yaml.safe_dump(ann_data, f)
            cap_right.release()
            os.remove(tmp_path_right)





def video_level_split(frames_root, output_split_file,train=0.8,val=0.1,seed=42):
    """
    Generate a video-level split mapping.
    Saves a YAML file with video IDs for train/val/test sets.
    """
    random.seed(seed)
    #frames_root = os.path.join(dataset_root, "Extracted2", "extracted_frames")
    videos = sorted(os.listdir(frames_root))
    random.shuffle(videos)

    n = len(videos)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        "train": videos[:n_train],
        "val": videos[n_train:n_train + n_val],
        "test": videos[n_train + n_val:]
    }

    with open(output_split_file, "w") as f:
        yaml.safe_dump(splits, f)

    print(f" Split file saved to: {output_split_file}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

def extract_corresponding_frames(video_split, new_frames_path, old_frames_path, dataset_root, set_type="test"):
    """
    Extract right-camera frames AND right keypoints matching the already-extracted left frames.

    video_split      : YAML file with train/val/test video IDs
    new_frames_path  : output folder to save right frames
    old_frames_path  : folder where left frames have already been extracted
    dataset_root     : root folder containing original zipped videos
    set_type         : "train" | "val" | "test"
    """

    num_keypoints = 14  # same as left side

    # Load list of video IDs
    with open(video_split, "r") as f:
        splits = yaml.safe_load(f)
    video_list = splits[set_type]

    # Map ZIP paths to video IDs
    zip_files = {
        os.path.splitext(os.path.basename(z))[0]: z
        for z in glob.glob(f"{dataset_root}/*.zip")
    }

    # Output directories
    if not os.path.exists(new_frames_path):
        os.makedirs(new_frames_path)

    kp_out_root = new_frames_path.replace("frames", "keypoints")
    if not os.path.exists(kp_out_root):
        os.makedirs(kp_out_root)

    print(f"Extracting RIGHT frames + keypoints for {set_type}: {len(video_list)} videos")

    for vid_id in tqdm(video_list):

        # Find corresponding ZIP file
        if vid_id not in zip_files:
            print(f"WARNING: ZIP file for {vid_id} not found.")
            continue

        zip_path = zip_files[vid_id]

        # Create output dirs
        vid_out_dir = os.path.join(new_frames_path, vid_id)
        vid_out_kp  = os.path.join(kp_out_root,  vid_id)
        os.makedirs(vid_out_dir, exist_ok=True)
        os.makedirs(vid_out_kp,  exist_ok=True)

        # Determine which frame numbers to extract
        left_frames_dir = os.path.join(old_frames_path, vid_id)
        left_frame_files = sorted(glob.glob(f"{left_frames_dir}/*.jpg"))

        if len(left_frame_files) == 0:
            print(f"WARNING: no extracted left frames found for {vid_id}")
            continue

        # Parse frame indices from filenames
        frame_indices = []
        for f in left_frame_files:
            fname = os.path.basename(f)
            idx = int(fname.split("_")[-1].split(".")[0])  
            frame_indices.append(idx)

        #  Load right video and right keypoints yaml from ZIP 
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:

            # Find right video file
            right_video_file = [f for f in zip_ref.namelist()
                                if f.endswith(".mp4") and "right" in f.lower()]
            if len(right_video_file) == 0:
                print(f"ERROR: No RIGHT video in {zip_path}")
                continue
            right_video_file = right_video_file[0]

            # Extract right video to temp file
            with zip_ref.open(right_video_file) as f:
                video_bytes = f.read()
            tmp_path = f"/tmp/{vid_id}_right_temp.mp4"
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(video_bytes)

            # Find right keypoint file
            right_kp_file = [f for f in zip_ref.namelist()
                             if "keypoints_right" in f.lower()]
            if len(right_kp_file) == 0:
                print(f"ERROR: No RIGHT keypoint YAML in {zip_path}")
                continue
            right_kp_file = right_kp_file[0]

            # Load YAML annotations
            with zip_ref.open(right_kp_file) as f:
                kp_data = yaml.safe_load(f)

            # Convert keys to int
            kp_data = {int(k): v for k, v in kp_data.items()}

        # Open right video file
        capR = cv2.VideoCapture(tmp_path)

        # --- Extract frames + keypoints ---
        for frame_idx in frame_indices:

            # Extract right frame
            capR.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            retR, frameR = capR.read()
            if not retR:
                print(f"Frame {frame_idx} missing in RIGHT for {vid_id}")
                continue

            save_frame_path = os.path.join(
                vid_out_dir, f"vid_{vid_id}_frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(save_frame_path, frameR)

            # Extract right keypoints
            if frame_idx not in kp_data:
                print(f"Missing RIGHT keypoints for frame {frame_idx} in video {vid_id}")
                continue

            frame_dict = kp_data[frame_idx]
            keypoints = []
            visibility = []

            for k in range(1, num_keypoints + 1):
                if k in frame_dict and frame_dict[k] is not None:
                    keypoints.append(frame_dict[k])
                    visibility.append(1)
                else:
                    keypoints.append([0, 0])
                    visibility.append(0)

            # Save keypoints YAML
            ann_path = os.path.join(
                vid_out_kp, f"vid_{vid_id}_frame_{frame_idx:06d}.yaml"
            )
            ann_data = {
                "video_id": vid_id,
                "frame_id": frame_idx,
                "keypoints": keypoints,
                "visibility": visibility
            }

            with open(ann_path, "w") as f:
                yaml.safe_dump(ann_data, f)

        capR.release()
        os.remove(tmp_path)

    print("Right-frame + keypoint extraction complete.")




if __name__=='__main__':
    #extract_frames()
    #old_frames_paths='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted/extracted_frames'
    #video_split_path ='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted/video_split.yaml'

    #video_level_split(frames_root,output_file, 0.5,0.5)
    #new_frames_path ='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_right_test/extracted_frames'
    #data_root = '/srv/homes/onbo10/thesis_Ons/SurgePoseData'
    #extract_corresponding_frames(video_split_path, new_frames_path, old_frames_paths,data_root)
    ####
    extract_frames_alternated()