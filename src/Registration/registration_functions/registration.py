
import numpy as np
import open3d as o3d
import copy

def register_cad_by_keypoints(mesh, cad_points, camera_points):
    """
    Method 1: Registers CAD mesh to camera space using the exact paired sequence
  
    """
    #  SVD Alignment Math
    centroid_cad = np.mean(cad_points, axis=0)
    centroid_cam = np.mean(camera_points, axis=0)

    A = cad_points - centroid_cad
    B = camera_points - centroid_cam

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_cam - R @ centroid_cad

    # Build rigid transform matrix and apply to millimeter mesh
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    result_mesh= mesh.copy()
    
    result_mesh.apply_transform(transformation_matrix)

    return result_mesh, transformation_matrix


def get_roi_from_skeleton_bounds(point_cloud, keypoints_list,colors=None, buffer=10.0):
    """
    Finds points that fall within an expanded bounding box of the skeleton keypoints.
    """
    pts = np.array(keypoints_list)
    min_bound = pts.min(axis=0) - buffer
    max_bound = pts.max(axis=0) + buffer
    
    # Create a mask for points inside the box
    mask = np.all((point_cloud >= min_bound) & (point_cloud <= max_bound), axis=1)
    if colors is not None:
        return point_cloud[mask], colors[mask]
    else:
        return point_cloud[mask]
        

def register_cad_to_pointcloud(initial_mesh, point_cloud_array, local_kpts_list, threshold = 2, buffer_zone=5.0, metrics_out= False):#threshold=2
    # Get ROI based on the skeleton of the already-registered tool
    local_point_cloud = get_roi_from_skeleton_bounds(point_cloud_array, local_kpts_list, buffer=buffer_zone)
    
    if len(local_point_cloud) < 200:
        print("Warning: ROI too sparse. Check skeleton alignment.")
        return initial_mesh, np.eye(4)

    #  Prepare O3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(initial_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(initial_mesh.faces)
    
   # Sample points from the mesh 
    cad_pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
    
    # Create target point cloud
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(local_point_cloud)
    
  
    reg_p2p = o3d.pipelines.registration.registration_icp(
        cad_pcd, 
        target_pcd, 
        threshold, 
        np.eye(4), # Initial state is identity
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) #
 
    
    # Check if it found a solution
    if reg_p2p.fitness < 0.1:
        print(f"Warning: ICP Fitness is low ({reg_p2p.fitness:.2f}). Check alignment.")

    result_mesh = initial_mesh.copy()
    result_mesh.apply_transform(reg_p2p.transformation)
    cad_pcd_transformed=copy.deepcopy(cad_pcd)
    cad_pcd_transformed.transform(reg_p2p.transformation)
    if metrics_out:
        return result_mesh, reg_p2p.transformation , reg_p2p.fitness, reg_p2p.inlier_rmse, target_pcd, cad_pcd, cad_pcd_transformed
    else:
        return result_mesh, reg_p2p.transformation

def apply_transformation_on_CAD_kpts(raw_kpts, tranformation):
    transformed_kpts={}
    for name, pos in raw_kpts.items():
        pos_cam = (tranformation @ np.append(np.array(pos), 1.0))[:3]
        transformed_kpts[name]=pos_cam
    return transformed_kpts