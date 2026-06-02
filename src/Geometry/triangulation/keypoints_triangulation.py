import os
import glob
import numpy as np
from src.Keypoints_detection.inference.inferencer import run_multi_tool_inference
from src.Geometry.triangulation.triangulation_utils import get_first_digit
import json



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


def triangulate_and_save_all(
    inferencer, 
    triangulator, 
    test_paths_l, 
    test_paths_r, 
    test_video_list, 
    org_dataset_path, 
    img_size,
    max_tools=2,
    save_path=''):
    print("Running batch inference on Test frames...")
    all_preds_l, all_masks_l = run_multi_tool_inference(inferencer, test_paths_l, max_tools)
    all_preds_r, all_masks_r = run_multi_tool_inference(inferencer, test_paths_r, max_tools)
    
    zip_files = sorted(glob.glob(f"{org_dataset_path}/*.zip"))
    frame_data_log = []

    # Process each video and frame
    for video_id in test_video_list:
        print(f"Processing Video: {video_id}")
        
        # Load calibration for this specific video
        zip_path = [f for f in zip_files if video_id in os.path.basename(f)][0]
        triangulator.load_calibration(zip_path)
        lmap1, lmap2, rmap1, rmap2, _,_ , p1, p2, r1, r2= triangulator.get_rectification_maps(img_size)
        # Get paths specific to this video
        vid_paths_l = [p for p in test_paths_l if get_first_digit(os.path.basename(p)) == video_id]
        vid_paths_r = [p for p in test_paths_r if get_first_digit(os.path.basename(p)) == video_id]

        for path_l, path_r in zip(vid_paths_l, vid_paths_r):
            # Extract the exact filename for the ID
            frame_id_str = os.path.basename(path_l).split('.')[0].replace('_left','')
            
            # Find the global index to pull corresponding predictions
            idx = test_paths_l.index(path_l)
            
            frame_entry = {
                'video_id': video_id,
                'frame_id_str': frame_id_str,
                'path_l': path_l,
                'path_r': path_r,
                'tools': []
            }

            for t in range(max_tools):
                # Isolate data for frame/tool
                p_l = all_preds_l[idx, t].reshape(1, -1, 2)
                p_r = all_preds_r[idx, t].reshape(1, -1, 2)
                m_l = all_masks_l[idx, t].reshape(1, -1)
                m_r = all_masks_r[idx, t].reshape(1, -1)
                
                #Rectify 2D points using rectified projection pi rectification rotation ri matrices (from get_rectification_maps)
                p_l_rect = triangulator.rectify_undistort_points(p1, r1, p_l, side='left')
                p_r_rect = triangulator.rectify_undistort_points(p2, r2, p_r, side='right')
                
                # Perform 3D triangulation
                pts_3d = triangulator.triangulate_rectified_kpts(p_l_rect, p_r_rect, m_l, m_r,p1,p2)

                # Record results
                frame_entry['tools'].append({
                    'tool_id': t,
                    'pts_3d': pts_3d.tolist(),
                    'preds_l': p_l.tolist(),
                    'preds_r': p_r.tolist()
                })
            
            frame_data_log.append(frame_entry)

    # Save to JSON log
    with open(save_path, 'w') as f:
        json.dump(frame_data_log, f, indent=4)
        
    print(f"Triangulation complete. Log saved to {save_path}")
    return frame_data_log


