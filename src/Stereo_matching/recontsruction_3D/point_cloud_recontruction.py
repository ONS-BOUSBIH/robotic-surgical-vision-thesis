from src.Segmentation.data_postprocessing.masks_postprocessing import * 
from src.Geometry.triangulation.triangulation_utils import *
import os
import numpy as np
from src.Stereo_matching.stereo_matching_utils import *
from src.Stereo_matching.postprocessing.outliers_3D_filtering import *


def filtered_point_clouds_from_npz(frame_filename, vid_id, npz_root, triangulator,rectification_mode = "conventional"):
    
    npz_path =  os.path.join(npz_root, vid_id, 'compressed_data', f"{frame_filename}.npz")
    left_mask_path, _, zip_calib_path, left_img_path, right_img_path, frame_id=get_file_paths_from_compressed_file_path(Path(npz_path),vid_id_position=1)
    data=np.load(npz_path)
    disparity=data['disparity']
    # load data
    mask_l = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)
    img_l = cv2.imread(left_img_path)
    img_r= cv2.imread(right_img_path)
    h, w = img_l.shape[:2]
    triangulator.load_calibration(zip_calib_path);
    lmap1, lmap2, rmap1, rmap2, q, r1, p1,p2,_,_= triangulator.get_rectification_maps(img_size=(h,w), mode=rectification_mode)
    rect_l, rect_r = triangulator.rectify_images(img_l, img_r,lmap1, lmap2, rmap1, rmap2, rectification_mode)
    rect_mask_l = cv2.remap(mask_l, lmap1, lmap2, cv2.INTER_NEAREST)
    eroded_mask= erode_mask(rect_mask_l, iterations=3)
    points_3d_cloud, colors,_ = triangulator.calculate_point_cloud_rectified(disparity, rect_l, eroded_mask,p1)
    # instruments_clouds_list,colors_list=get_filtered_and_ordered_clusters(points_3d_cloud,colors, eps=5, min_points=20, top_n=2)
    instruments_clouds_list,colors_list=get_filtered_and_ordered_clusters_fast(points_3d_cloud,colors, eps=5, min_points=20, top_n=2)
    return instruments_clouds_list,colors_list
