import plotly.graph_objects as go
import numpy as np
import json
import open3d as o3d


def plot_mesh_and_keypoints(mesh,keypoints, kpts_colors):
    fig = go.Figure()
    color_visuals = mesh.visual.to_color()
    v_colors = color_visuals.vertex_colors
    colors = v_colors[:, :3] / 255.0
# Add the Mesh trace
    fig.add_trace(go.Mesh3d(
    x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
    vertexcolor=colors, opacity=1, name='LND Mesh'
))
# Add the Keypoints
    for name, pos in keypoints.items():

        fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=10, color=kpts_colors[f'{name}'], symbol='circle'),
        text=[f'{name}'],
        textposition="top center",
        name=f'{name}'
    ))

    fig.update_layout(
    scene=dict(aspectmode='data'),
    title="Final Model Keypoints for Registration",
    width=1000, height=800
)

    fig.show()

def plot_colored_mesh_plotly(mesh, title="LND Colored Model"):
    v = mesh.vertices
    f = mesh.faces
    
    # Get colors from trimesh
    if hasattr(mesh.visual.to_color(), 'vertex_colors'):
        color_visuals = mesh.visual.to_color()
        v_colors = color_visuals.vertex_colors
        colors = v_colors[:, :3] / 255.0


    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            vertexcolor=colors, 
            opacity=1.0,
            flatshading=False 
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        width=900, height=700
    )
    fig.show()



def load_tool_data(filepath, part_name):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    part_data = data.get(part_name)
    if not part_data:
        raise ValueError(f"Part '{part_name}' not found in {filepath}")
    
    return part_data['keypoints'], part_data['colors']


def plot_registration_overlay(new_mesh_left,new_mesh_right, raw_kpts_left, raw_kpts_right,raw_kpts_color_dict, camera_kpts_left, camera_kpts_right, 
                                            cloud_points_left=None, cloud_points_right=None,cloud_colors_left=None, cloud_colors_right=None):
    fig = go.Figure()
  
    def swap(pts):
            return np.column_stack((pts[:, 0], pts[:, 2], -pts[:, 1]))
    all_pts = []
    for i, new_meshes in enumerate([new_mesh_left,new_mesh_right]):
        for new_mesh in new_meshes:
            v = np.asarray(new_mesh.vertices)
            f = np.asarray(new_mesh.faces)
            v_swapped = swap(v)
            
            try:
                mesh_colors = new_mesh.visual.to_color().vertex_colors[:, :3] / 255.0
            except:
                mesh_colors = np.full((len(v), 3), [0.85, 0.85, 0.85])

            fig.add_trace(go.Mesh3d(
                x=v_swapped[:, 0], y=v_swapped[:, 1], z=v_swapped[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                vertexcolor=mesh_colors, name='Registered CAD Mesh'
            ))
            all_pts.append(v_swapped)
    
    cloud_colors=[cloud_colors_left,cloud_colors_right]
    for i, cloud_points in enumerate([cloud_points_left, cloud_points_right]):   
        if cloud_points is not None:
            c_swapped = swap(cloud_points)
            rgb = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in cloud_colors[i]]
            fig.add_trace(go.Scatter3d(
                    x=c_swapped[:, 0], y=c_swapped[:, 1], z=c_swapped[:, 2],
                    mode='markers', marker=dict(size=1, color=rgb, opacity=0.5), name='S2M2 Cloud'
                ))
            all_pts.append(c_swapped)



    for i, raw_kpts_sets in enumerate([raw_kpts_left,raw_kpts_right]):
        for raw_kpts in raw_kpts_sets:
            for name, pos_cam in raw_kpts.items():
                # Transformation and Swapping
                pos_swapped = swap(pos_cam.reshape(1, 3))
                marker_color = raw_kpts_color_dict.get(name, "grey")
                fig.add_trace(go.Scatter3d(
                    x=pos_swapped[:, 0], 
                    y=pos_swapped[:, 1], 
                    z=pos_swapped[:, 2],
                    mode='markers', 
                    marker=dict(
                        size=5, 
                        color=marker_color, 
                        symbol='circle', 
                        line=dict(color='white', width=1)
                    ), 
                    name=f"CAD: {name}"
                ))
                all_pts.append(pos_swapped)

    
    # Swap Camera Keypoints for Skeleton
    for camera_kpts in [camera_kpts_left,camera_kpts_right]:
        cam_swapped = swap(camera_kpts)
        
        # Skeleton Edges
        defined_edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
        for start, end in defined_edges:
            fig.add_trace(go.Scatter3d(
                x=[cam_swapped[start, 0], cam_swapped[end, 0]],
                y=[cam_swapped[start, 1], cam_swapped[end, 1]],
                z=[cam_swapped[start, 2], cam_swapped[end, 2]],
                mode='lines', line=dict(color='darkorange', width=5), showlegend=False
            ))

        fig.add_trace(go.Scatter3d(
            x=cam_swapped[:, 0], y=cam_swapped[:, 1], z=cam_swapped[:, 2],
            mode='markers+text', text=[f"P{i+1}" for i in range(7)],
            marker=dict(size=6, color='darkorange'), name='Joints'
        ))
        all_pts.append(cam_swapped)
    
    all_pts = np.vstack(all_pts)
    
    fig.update_layout( 
    scene=dict(
        aspectmode='data',
        xaxis=dict(range=[all_pts[:,0].min(), all_pts[:,0].max()]),
        yaxis=dict(range=[all_pts[:,1].min(), all_pts[:,1].max()]),
        zaxis=dict(range=[all_pts[:,2].min(), all_pts[:,2].max()])
    ))
    fig.show()



def get_ordered_instrument_clusters_and_colors(point_cloud_array, colors, cluster_tolerance=10.0, min_points=100):
    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array)

    # Euclidean Clustering
    labels = np.array(pcd.cluster_dbscan(eps=cluster_tolerance, min_points=min_points))
    
    # Filter noise and prepare for sorting
    valid_mask = labels != -1
    labels = labels[valid_mask]
    pts = point_cloud_array[valid_mask]
    cols = np.array(colors)[valid_mask]
    
    unique_labels = np.unique(labels)
    
    #Calculate centroids for all clusters at once
    label_counts = np.bincount(labels)
    label_sums_x = np.bincount(labels, weights=pts[:, 0])
    centroids_x = label_sums_x[unique_labels] / label_counts[unique_labels]
    
    # Sort labels based on centroid X
    sorted_idx = np.argsort(centroids_x)
    
    # Extract clusters in order
    sorted_clusters = []
    sorted_colors = []
    
    for label in unique_labels[sorted_idx]:
        mask = (labels == label)
        sorted_clusters.append(pts[mask])
        sorted_colors.append(cols[mask])

    return sorted_clusters, sorted_colors