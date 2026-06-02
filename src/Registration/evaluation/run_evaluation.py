import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
   
import trimesh
from src.Geometry.triangulation.triangulation_utils import get_frame_data
from src.Geometry.triangulation.triangulator import Triangulator
import numpy as np
from src.Stereo_matching.recontsruction_3D.point_cloud_recontruction import filtered_point_clouds_from_npz
from src.Registration.registration_functions.registration import *
from src.Registration.registration_functions.registration_utils import *
from src.Registration.evaluation.evaluation import *
import argparse
import yaml
from tqdm import tqdm
import csv

def run_evaluation_one_instance(vid_id, frame_name, predicted_kpts_path, mesh_path, mesh_kpts_file_path, disparity_map_folder_root, part_name, buffer_zone, kpts_idx_list=[1,2,5]):
    
    # Get the Keypoints predictions 
    result = get_frame_data(predicted_kpts_path, vid_id, frame_name)
        # Extract and convert each tool's points to a numpy array
    pts_tool_0 = np.array(result['tools'][0]['pts_3d'])
    pts_tool_1 = np.array(result['tools'][1]['pts_3d'])
        #Concatenate the results of both tools
    pts_3d = np.vstack([pts_tool_0, pts_tool_1])
    
    # Initiate the triangulator instance
    tri = Triangulator(num_keypoints=7)
    
    # Get the point cloud reconstruction for each instrument from the disparity map prediction
    instruments_clouds,_= filtered_point_clouds_from_npz(frame_filename=frame_name, vid_id=vid_id, npz_root=disparity_map_folder_root, triangulator=tri)
    
    #Load the 3D model mesh and scale it to meters
    
    model_3d_mesh = trimesh.load(mesh_path, force='mesh')
    model_3d_mesh.apply_scale(1000)
    model_3d_kpts, model_3d_colors=load_tool_data(mesh_kpts_file_path, part_name=part_name)

    model_3d_mesh_left = model_3d_mesh.copy()
    model_3d_mesh_right = model_3d_mesh.copy()
    
    # Extract 3D model keypoints from dict
    cad_points= []
    for name, pos in model_3d_kpts.items():
        cad_points.append(model_3d_kpts[name])
    cad_points=np.array(cad_points)

    #Extract matching camera targets keypoints 
    camera_points_left=[]
    camera_points_right=[]
    for i in kpts_idx_list:
        camera_points_left.append(pts_3d[0][i])
        camera_points_right.append(pts_3d[1][i])
    camera_points_left=np.array(camera_points_left)
    camera_points_right=np.array(camera_points_right) 

    # Register the 3D model to the keypoint skeleton
    registered_model_3d_left, T_matrix_left = register_cad_by_keypoints(model_3d_mesh_left, cad_points, camera_points_left)
    registered_model_3d_right, T_matrix_right = register_cad_by_keypoints(model_3d_mesh_right, cad_points, camera_points_right)
    
    # Apply the same transformation to the mesh's keypoints
    model_3d_cad_kpts_left= apply_transformation_on_CAD_kpts(model_3d_kpts,T_matrix_left)
    model_3d_cad_kpts_right= apply_transformation_on_CAD_kpts(model_3d_kpts,T_matrix_right)
    
    # Register the 3D model to the point cloud usisng the last registration state as initialisation

    model_3d_icp_left, T_icp_left, left_fitness, left_rmse, left_target_pcd, left_cad_mesh=register_cad_to_pointcloud(registered_model_3d_left, 
                                                                                                                      instruments_clouds[0], camera_points_left, metrics_out=True,buffer_zone=buffer_zone)
    model_3d_icp_right, T_icp_right, right_fitness, right_rmse, right_target_pcd, right_cad_mesh=register_cad_to_pointcloud(registered_model_3d_right, 
                                                                                                                            instruments_clouds[1], camera_points_right, metrics_out=True,buffer_zone=buffer_zone)
    
    model_3d_icp_kpts_left= apply_transformation_on_CAD_kpts(model_3d_cad_kpts_left,T_icp_left)
    model_3d_icp_kpts_right= apply_transformation_on_CAD_kpts(model_3d_cad_kpts_right,T_icp_right)
    
    # Get the error measures between the keypoints from registration method 1 and the predicted keypoints
    tre_left_1=calculate_tre(model_3d_cad_kpts_left, camera_points_left)
    tre_right_1=calculate_tre(model_3d_cad_kpts_right, camera_points_right)
    
    # Get the error measures between the keypoints from registration method 2 and the predicted keypoints
    tre_left_2=calculate_tre(model_3d_icp_kpts_left, camera_points_left)
    tre_right_2=calculate_tre(model_3d_icp_kpts_right, camera_points_right)
    
    # Compute the Chamfer distance between the registered mesh and the adequte point cloud surface
    cd_left=calculate_chamfer_distance(left_cad_mesh,left_target_pcd)
    cd_right=calculate_chamfer_distance(right_cad_mesh,right_target_pcd)

    return tre_left_1, tre_right_1, cd_left, cd_right, tre_left_2, tre_right_2, left_fitness, right_fitness, left_rmse, right_rmse


def run_evaluation_test_set(videos_list, predicted_kpts_path, mesh_path, mesh_kpts_file_path, disparity_map_folder_root, output_path, part_name='pitch_link', buffer_zone=6.0, kpts_idx_list=[1, 2, 5]):
    root_path = Path(disparity_map_folder_root)
    output_file = Path(output_path)
    
    # Define header for the CSV
    header = [
            'vid_id', 'frame_name', 
            'Mean_TRE_L1', 'RMSE_TRE_L1', 'Mean_TRE_R1', 'RMSE_TRE_R1', # Method 1
            'Mean_TRE_L2', 'RMSE_TRE_L2', 'Mean_TRE_R2', 'RMSE_TRE_R2', # Method 2
            'Chamfer_L', 'Chamfer_R',
            'Fitness_L', 'Fitness_R',
            'RMSE_ICP_L', 'RMSE_ICP_R'
            ]

    # Create file and write header if it doesn't exist
    if not output_file.exists():
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for vid_id in tqdm(videos_list, unit='video'):
        video_folder = root_path / vid_id / "compressed_data"
        
        if not video_folder.exists():
            continue

        frame_files = sorted(list(video_folder.glob("*.npz")))
        
        for frame_file in tqdm(frame_files, desc=f'Processing {vid_id}', unit='frame'):
            frame_name = frame_file.stem 
            
            try:
                tre_left1, tre_right1, cd_left, cd_right, tre_left2, tre_right2, fit_left, fit_right, rmse_left, rmse_right = run_evaluation_one_instance(
                    vid_id=vid_id,
                    frame_name=frame_name,
                    predicted_kpts_path=predicted_kpts_path,
                    mesh_path=mesh_path,
                    mesh_kpts_file_path=mesh_kpts_file_path,
                    disparity_map_folder_root=disparity_map_folder_root,
                    kpts_idx_list=kpts_idx_list, part_name=part_name, buffer_zone=buffer_zone
                )
                
                # Append row immediately
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                                    vid_id, frame_name,
                                    tre_left1[0], tre_left1[1], tre_right1[0], tre_right1[1],  # Method 1 errors
                                    tre_left2[0], tre_left2[1], tre_right2[0], tre_right2[1],  # Method 2 errors
                                    cd_left, cd_right,                                  # Chamfer Distances
                                    fit_left, fit_right,                                # Fitness
                                    rmse_left, rmse_right                               # ICP Inlier RMSE
                                ])
            except Exception as e:
                print(f"Failed to process {frame_name}: {e}")

    print(f"Evaluation finished. Results updated in {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the inference config file')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        config_data = yaml.safe_load(f)
    
    videos_list= config_data['LND_test_videos']
    predicted_kpts_path= config_data['kpts_predictions_folder']
    mesh_path= config_data['3D_model_path']
    mesh_kpts_file_path= config_data['3D_model_kpts_path']
    disparity_map_folder_root= config_data['disparity_map_predictions_folder']
    output_path= config_data['registration_results_path']
    kpts_idx_list= config_data['indices_of_registration_kpts_in_prediction']
    tool_part_name= config_data['3D_model_part_name']
    buffer_zone=config_data['point_cloud_buffer_zone']

    run_evaluation_test_set(videos_list, predicted_kpts_path, mesh_path, mesh_kpts_file_path, 
                            disparity_map_folder_root, output_path, part_name=tool_part_name, buffer_zone=buffer_zone, kpts_idx_list=kpts_idx_list)

if __name__=='__main__':
    main()