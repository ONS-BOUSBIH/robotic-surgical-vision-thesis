
import numpy as np
import open3d as o3d

def DBSCAN_cluster_filtering( points_np, eps=5, min_points=20, top_n=2):
    """
    Uses DBSCAN clustering algorithm and removes 3D outlier points clusters by keeping the largest 2 clusters (for 2 surgical tools).
    Input: points_np (N, 3) numpy array
    Output: filtering mask (N,) numpy array
    """
    if points_np.shape[0] == 0:
        return points_np

    #Convert NumPy array to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    #Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if len(labels) == 0 or np.all(labels == -1):
            
        return np.array([]) # Return empty if everything is noise

    #Find the largest clusters
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
    # Sort by count descending and take top_n
    top_cluster_indices = unique_labels[np.argsort(counts)[::-1][:top_n]]
        
    #Create mask for points belonging to the cluster
    mask = np.isin(labels, top_cluster_indices)
        
    return mask
    
def adaptive_z_filtering(points_np, percentile=97):
    """
    Isolates foreground objects by generating a boolean mask for points within a dynamic depth threshold calculated via percentiles.
    Input: points_np (np.ndarray): (N, 3) array of point cloud coordinates (X, Y, Z).
            percentile (float): The percentage of closest points to retain (0-100).
    Output: np.ndarray: A boolean mask where True indicates points that passed the depth filter.
    """
    # Assuming teh tools are the closest objects (True for Surgpose)
    z_values = points_np[:, 2]
    # Calculate a threshold that keeps the closest X% of points
    z_threshold = np.percentile(z_values, percentile)
    mask = z_values <= z_threshold
    return mask

    



