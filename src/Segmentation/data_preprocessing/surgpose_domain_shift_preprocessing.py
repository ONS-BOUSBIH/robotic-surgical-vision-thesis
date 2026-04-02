import cv2
import numpy as np
import json
import os

def normalize_to_endovis_fingerprint(img_rgb, stats):
    img_float = img_rgb.astype(np.float32)
    norm_img = np.zeros_like(img_float)
    
    for i in range(3):
        current_mean = np.mean(img_float[:,:,i])
        current_std = np.std(img_float[:,:,i])
        
        target_mean = stats[i][0]
        target_std = stats[i][1]
        
        norm_img[:,:,i] = ((img_float[:,:,i] - current_mean) / (current_std + 1e-5)) * target_std + target_mean
    
    return np.clip(norm_img, 0, 255).astype(np.uint8)

def process_dataset(json_path, input_path, output_path):
    # 1. Load the JSON fingerprint
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    props = data["foreground_intensity_properties_per_channel"]
    target_stats = {int(k): (v["mean"], v["std"]) for k, v in props.items()}

    os.makedirs(output_path, exist_ok=True)

    # Filter for png files
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith('.png')]

    print(f"Found {len(image_files)} images in {input_path}")

    for img_name in image_files:
        # FULL PATH to the source image
        full_input_path = os.path.join(input_path, img_name)
        
        img_bgr = cv2.imread(full_input_path)
        if img_bgr is None:
            print(f"Failed to read: {full_input_path}")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        normalized_rgb = normalize_to_endovis_fingerprint(img_rgb, target_stats)
        final_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
        
        # FULL PATH to the destination
        full_output_path = os.path.join(output_path, img_name)
        cv2.imwrite(full_output_path, final_bgr)

    print(f"Done! Normalized images saved to: {output_path}")

if __name__ == "__main__":
    process_dataset(
        json_path='results/Segmentation/nnunet_preprocessed/Dataset789_Endovis17/dataset_fingerprint.json', 
        input_path='/srv/homes/onbo10/thesis_main/data/EndoVis2017/Surgpose_for_nnUnet_test/imagesTs', 
        output_path='/srv/homes/onbo10/thesis_main/data/EndoVis2017/Surgpose_for_nnUnet_test_color_normalized/imagesTs'
    )