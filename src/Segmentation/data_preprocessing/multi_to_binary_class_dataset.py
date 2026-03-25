import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def binarize_masks(root_dir,target_dir,target_folders):
    for folder in target_folders:
        folder_path = os.path.join(root_dir, folder)
        target_folder_path =os.path.join(target_dir,folder)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder}: Folder not found.")
            continue
       
        print(f"Processing {folder}...")
        # Create the target path if not exists 
        os.makedirs(target_folder_path, exist_ok=True)
        # List all png files
        mask_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        for filename in tqdm(mask_files):
            file_path = os.path.join(folder_path, filename)
            target_file_path = os.path.join(target_folder_path, filename)
            # Open image and convert to numpy array
            with Image.open(file_path) as img:
                mask = np.array(img)
            
            # Binary logic: foreground Vs Background
            binary_mask = (mask > 0).astype(np.uint8)
            
            #Save the binary mask
            result_img = Image.fromarray(binary_mask)
            result_img.save(target_file_path)

if __name__ == "__main__":
    
    orig_dataset_path = "./data/EndoVis2017/Dataset704_Endovis17"
    target_dataset_path = "./data/EndoVis2017/Dataset789_Endovis17"
    folders= ['labelsTr','labelsTs']
    binarize_masks(root_dir=orig_dataset_path, target_dir=target_dataset_path, target_folders=folders)
    print("Done!")