
import cv2
import os
import numpy as np
from src.Stereo_matching.stereo_matching_utils import *
from src.Geometry.triangulation.triangulator import Triangulator
from src.Segmentation.data_postprocessing.masks_postprocessing import *
import csv
import argparse
import yaml
from tqdm import tqdm

def disparity_segmentatiom_masks_comparison(disp_root, file_path_structure, csv_path, img_shape,mask_postprocess,rect_mode="conventional"):
    
    #get all the saved compressed file paths
    disp_paths= get_compressed_paths_list(disp_root, file_path_structure)
    h, w = img_shape
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_headers = ['Frame', 'Disparity threshold', 'Hausdorff distance', 'Dice score']
        
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    for file in tqdm(disp_paths, desc='Mask evaluation', unit='frame'):
        #Load the precomputed disparity
        data=np.load(file)
        disp= data['disparity']
        
        # Create a disparity mask, load and rectify segmenatation mask, compute Hausdorff distance and dice score between both masks
        disp_thres,_ , _, frame_id, hd, dice = disparity_vs_segmentation(mask_postprocess, rect_mode, h, w, file, disp)

        #save metrics 
        stats = {
                'Frame': frame_id,
                'Disparity threshold': round(disp_thres, 4),
                'Hausdorff distance': round(hd, 4),
                'Dice score': round(dice, 4),
                }
                
        with open(csv_path, mode='a', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_headers).writerow(stats)

def disparity_vs_segmentation(mask_postprocess, rect_mode, h, w, disp_file, disp):
    
    # Create a disparity mask using thresholding
    disp_thres, disp_mask= from_disparity_to_mask(disparity_map=disp)

    # get relevant files paths for segmentation mask
    left_mask_path, _ , zip_calib_path, left_img_path, right_img_path, frame_id= get_file_paths_from_compressed_file_path(disp_file,vid_id_position=1)
    video_id = disp_file.parents[1].name 
        #load calibration and rectify
        
    triangulator = Triangulator()
    triangulator.load_calibration(zip_calib_path);
        
    mask_l = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)
    img_l = cv2.imread(left_img_path)
    img_r= cv2.imread(right_img_path)
        
    lmap1, lmap2, rmap1, rmap2, q= triangulator.get_rectification_maps(img_size=(h,w), mode=rect_mode)
    rect_l, rect_r = triangulator.rectify_images(img_l, img_r,lmap1, lmap2, rmap1, rmap2, rect_mode)
    rect_mask_l = cv2.remap(mask_l, lmap1, lmap2, cv2.INTER_NEAREST)
    if mask_postprocess:
        rect_mask_l= filter_binary_mask(rect_mask_l)
        disp_mask= filter_binary_mask(disp_mask)
        #compute metrics
    hd= compute_hausdorf_distance(disp_mask,rect_mask_l)
    dice= compute_dice_score(disp_mask,rect_mask_l)
    return disp_thres, disp_mask,rect_mask_l,frame_id,hd,dice

def run_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        config_data = yaml.safe_load(f)
    
    disparity_segmentatiom_masks_comparison(config_data['disparity_root'], config_data['file_path_structure'], config_data['csv_path'], img_shape= (config_data['h'],config_data['w']),mask_postprocess=config_data['postprocessing'])
    
    

if __name__ == "__main__":
    run_comparison()





















        

