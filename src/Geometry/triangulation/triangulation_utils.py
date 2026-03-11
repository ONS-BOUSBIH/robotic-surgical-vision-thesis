import numpy as np
import re
import glob 
import yaml 
import os

def get_first_digit(string):
    match = re.search(r"\d+", string)
    return match.group() if match else None


def get_paths_and_video_lists(data_root_l, data_root_r, split_file):
    with open(split_file, "r") as f:
        splits = yaml.safe_load(f)
    test_video_list = splits['test']
    
    # Get all paths 
    all_img_paths_l = sorted(glob.glob(f'{data_root_l}/**/*.jpg', recursive=True))
    all_img_paths_r = sorted(glob.glob(f'{data_root_r}/**/*.jpg', recursive=True))

    # Only keep paths where the video ID is in test_video_list
    test_paths_l = [p for p in all_img_paths_l if get_first_digit(os.path.basename(p)) in test_video_list]
    test_paths_r = [p for p in all_img_paths_r if get_first_digit(os.path.basename(p)) in test_video_list]

    print(f"Total images in directory: {len(all_img_paths_l)}")
    print(f"Images to process (Test Split): {len(test_paths_l)}")
    return test_video_list,test_paths_l,test_paths_r

def calculate_success_metrics(results_dict):
    """
    Calculates the detection success rate for each tool in the results.
    """
    success_stats = {}
    
    for t_idx in range(len(results_dict['tri_3d'])):
        # List of 3D point arrays, one array per video
        all_videos_pts = results_dict['tri_3d'][t_idx]
        
        total_frames = 0
        valid_frames = 0
        
        for video_pts in all_videos_pts:
            # video_pts shape: (Frames, 7, 3)
            total_frames += len(video_pts)
            
            # A frame is "successful" if the 3D point is not NaN
            # We check the first keypoint's X coordinate as a proxy
            is_valid = ~np.isnan(video_pts[:, 0, 0])
            valid_frames += np.sum(is_valid)
            
        success_rate = (valid_frames / total_frames) * 100 if total_frames > 0 else 0
        success_stats[f'Tool {t_idx}'] = {
            'Total Frames': total_frames,
            'Successful Detections': valid_frames,
            'Success Rate (%)': round(success_rate, 2)
        }
        
    return success_stats
import random
import os

def get_random_stereo_pairs(left_paths, right_paths, num_pairs=5, seed=42):
    """
    Creates a list of randomly selected [left, right] pairs from ordered lists.
    Returns: A list of dicts containing index, paths, and the source video ID 
             using the project's get_first_digit logic.
    """
    # 1. Set the seed for reproducibility
    random.seed(seed)
    
    # 2. Safety check: ensure lists are the same length
    if len(left_paths) != len(right_paths):
        print(f"Warning: List mismatch ({len(left_paths)} vs {len(right_paths)}). Truncating.")
        min_len = min(len(left_paths), len(right_paths))
        left_paths, right_paths = left_paths[:min_len], right_paths[:min_len]

    # 3. Zip with original indices
    indexed_pairs = list(enumerate(zip(left_paths, right_paths)))
    
    # 4. Sample
    sample_size = min(num_pairs, len(indexed_pairs))
    sampled_data = random.sample(indexed_pairs, sample_size)
    
    # 5. Format output
    results = []
    for idx, (l_p, r_p) in sampled_data:
        # Using your specific helper to extract the video ID
        v_id = get_first_digit(os.path.basename(l_p))
        
        results.append({
            'index': idx,
            'paths': [l_p, r_p],
            'video_id': v_id
        })
        
    return results