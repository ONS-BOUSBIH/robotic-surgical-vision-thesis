import os
import glob
import numpy as np
from src.Keypoints_detection.inference.inferencer import run_multi_tool_inference
from src.Geometry.triangulation.triangulation_utils import get_first_digit



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

