
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

def get_filtered_and_ordered_clusters(points_np, colors, eps=5.0, min_points=20, top_n=2):
    """
    Performs DBSCAN cluster filtering and the ordered clusters/colors.
    """
    if points_np.shape[0] == 0:
        return np.array([]), [], []

    #Clustering 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    #Filter for top_n largest clusters
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return np.zeros(len(points_np), dtype=bool), [], []

    # Get top_n cluster indices
    top_cluster_indices = unique_labels[np.argsort(counts)[::-1][:top_n]]
    
    # Create mask for filtering
    mask = np.isin(labels, top_cluster_indices)
    
    # Extract only the data for these top clusters
    filtered_labels = labels[mask]
    filtered_pts = points_np[mask]
    filtered_cols = np.array(colors)[mask]

    #Order clusters by X-centroid
    #Get centroids of the filtered clusters
    unique_filtered_labels = np.unique(filtered_labels)
    centroids_x = []
    for lbl in unique_filtered_labels:
        centroids_x.append(np.mean(filtered_pts[filtered_labels == lbl, 0]))
    
    # Sort labels based on X-centroid
    sorted_idx = np.argsort(centroids_x)
    sorted_labels = unique_filtered_labels[sorted_idx]

    # Organize output
    sorted_clusters = []
    sorted_colors = []
    for label in sorted_labels:
        cluster_mask = (filtered_labels == label)
        sorted_clusters.append(filtered_pts[cluster_mask])
        sorted_colors.append(filtered_cols[cluster_mask])

    return sorted_clusters, sorted_colors

    

def get_filtered_and_ordered_clusters_fast(points_np, colors, eps=5.0, min_points=20, top_n=2, voxel_size=1.0):
    """
    Downsamples for fast clustering, then projects labels back to original points.
    """
    if points_np.shape[0] == 0:
        return [], []

    # Downsample 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts_down = np.asarray(pcd_down.points)
    
    #Fast Clustering on downsampled data
    labels_down = np.array(pcd_down.cluster_dbscan(eps=eps, min_points=min_points))
    
    #Filter top_n labels from downsampled result
    unique_labels, counts = np.unique(labels_down[labels_down >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return [], []
    top_labels = unique_labels[np.argsort(counts)[::-1][:top_n]]
    
    #Project labels to original points using a KDTree
    #This finds the nearest downsampled point for every single original point
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)
    full_labels = np.full(points_np.shape[0], -1, dtype=int)
    
    for i in range(points_np.shape[0]):
        # Search for the 1 nearest neighbor in the downsampled cloud
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points_np[i], 1)
        nearest_down_label = labels_down[idx[0]]
        if nearest_down_label in top_labels:
            full_labels[i] = nearest_down_label

    #Extract and Sort full-res clusters by X-centroid
    sorted_clusters = []
    sorted_colors = []
    
    cluster_data = []
    for label in top_labels:
        mask = (full_labels == label)
        cluster_pts = points_np[mask]
        cluster_cols = np.array(colors)[mask]
        centroid_x = np.mean(cluster_pts[:, 0])
        cluster_data.append((centroid_x, cluster_pts, cluster_cols))
    
    cluster_data.sort(key=lambda x: x[0])
    
    for _, pts, cols in cluster_data:
        sorted_clusters.append(pts)
        sorted_colors.append(cols)

    return sorted_clusters, sorted_colors

