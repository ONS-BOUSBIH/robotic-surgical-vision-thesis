import numpy as np
import open3d as o3d

def calculate_tre(cad_kpts_transformed, camera_kpts_gt):
    """
    Target Registration Error 
    cad_kpts_transformed: Dict {name: array([x, y, z]), ...}
    camera_kpts_gt: ordered List of arrays [array([x, y, z]), ...] 
    """
    # Convert Dicts to sorted lists based on keys to ensure order matches
    keys = sorted(cad_kpts_transformed.keys())
    # Extract points in the same order
    cad_pts = np.array([cad_kpts_transformed[k] for k in keys])
    gt_pts = np.array(camera_kpts_gt)
    
    errors = np.linalg.norm(cad_pts - gt_pts, axis=1)
    return np.mean(errors), np.sqrt(np.mean(errors**2))

def calculate_chamfer_distance(mesh_pcd, target_pcd):
    """
    Returns the average Chamfer Distance.
    """
    #Compute distances 
    dists_mesh_to_target = np.asarray(mesh_pcd.compute_point_cloud_distance(target_pcd))
    dists_target_to_mesh = np.asarray(target_pcd.compute_point_cloud_distance(mesh_pcd))
    
    #Average the two directional distances
    cd = (np.mean(dists_mesh_to_target) + np.mean(dists_target_to_mesh)) / 2
    return cd

