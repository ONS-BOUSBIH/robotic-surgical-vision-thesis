import json
import os
import cv2
import numpy as np
import glob
import yaml
# FOR VITPOSE TRAINING
CATEGORIES = [{
    "id": 1,
    "name": "surgical_instrument",
    "keypoints": ["shaft", "wrist_pivot_1", "wrist_pivot_2", 
        "clasper_tip_1", "clasper_tip_2", "redundant_wrist_mark", "redundant_housing_mark"],
    "skeleton": [[1, 2], [2, 3], [3, 4], [3, 5]] 
    }]

def is_valid_sample(joints, vis, bbox):
  
    x1, y1, x2, y2 = map(int, bbox)
    bw = x2 - x1
    bh = y2 - y1

    #bbox size check
    if bw < 20 or bh < 20:
        return False

    #bbox area check
    if bw * bh < 400:
        return False

    #keypoints inside bbox
    for (x, y), v in zip(joints, vis):
        if v > 0:
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                return False
    return True

def convert_to_coco(image_folder, label_folder, output_json, vid_ids):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES 
    }
    
    ann_id = 0
    image_id = 0
    
    # Sort to keep training/validation consistent
    img_paths=[]
    for vid in vid_ids:
        vid_path= os.path.join(image_folder,vid)
        if not os.path.exists(vid_path):
            continue
        frames=[os.path.join(vid_path,f) for f in os.listdir(vid_path) if f.endswith('.jpg')]
        img_paths.extend(frames) 
    img_paths=sorted(img_paths)

    num_cor=0
    for img_path in img_paths:
        # 1. Load Image for dimensions
        img_name= os.path.basename(img_path)
        vid_id = os.path.basename(os.path.dirname(img_path))
        relative_path = os.path.join(vid_id, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": relative_path,
            "height": h,
            "width": w
        })
        
        yaml_name = img_name.replace('.jpg', '.yaml')
        yaml_path = os.path.join(label_folder, vid_id, yaml_name)
        
        if not os.path.exists(yaml_path):
            image_id += 1
            continue

        with open(yaml_path, 'r') as f:
            label_data = yaml.safe_load(f)
        
       
        #Process each instrument in the image
        for obj in label_data.get('objects', []):
            raw_kpts = obj['keypoints']     
            visibility = obj['visibility']    
            bbox=obj['bbox']
            
            if not is_valid_sample(raw_kpts,visibility,bbox):
                num_cor+=1
                continue
            # Convert to COCO flat format: [x, y, v, x, y, v...]
            # Map visibility flag 1 to COCO's 2 (visible and labeled)
            coco_kpts = []
            for i in range(len(raw_kpts)):
                x, y = raw_kpts[i]
                v = 2 if visibility[i] > 0 else 0
                coco_kpts.extend([float(x), float(y), v])
            
            # Handle BBox
            # From [x1, y1, x2, y2] Convert to COCO's format  [x_min, y_min, width, height]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            
            # Add 5% padding to the bbox
           
            padding_w = bw * 0.05
            padding_h = bh * 0.05
            
            x1_clipped = max(0,x1- padding_w)
            y1_clipped= max(0, y1- padding_h)
            x2_clipped= min(w-1,x2 + padding_w)
            y2_clipped= min(h-1,y2 + padding_h)

            final_w= x2_clipped- x1_clipped
            final_h= y2_clipped - y1_clipped

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": coco_kpts,
                "num_keypoints": int(sum(1 for v in visibility if v > 0)),
                "bbox": [float(x1_clipped), float(y1_clipped), 
                         float(final_w), float(final_h)],
                "area": float(bw * bh),
                "iscrowd": 0
            })
            ann_id += 1
            
        image_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Successfully created {output_json} with {ann_id} annotations, num_corrupt: {num_cor}.")

def main():
    
    split_path='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/video_split.yaml'
    with open(split_path,'r') as f:
        splits=yaml.safe_load(f)
    train_vid_list=splits['train']
    val_vid_list=splits['val']
    test_vid_list =splits['test']
    img_folder='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_frames'
    lbl_folder='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_bboxes_kpts'
    
    train_output = '/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/train.json'
    val_output = '/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/val.json'
    test_output='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/test.json'
    # convert_to_coco(image_folder=img_folder, label_folder=lbl_folder, output_json=train_output, vid_ids=train_vid_list)
    # convert_to_coco(image_folder=img_folder, label_folder=lbl_folder, output_json=val_output, vid_ids=val_vid_list)
    convert_to_coco(image_folder=img_folder, label_folder=lbl_folder, output_json=test_output, vid_ids=test_vid_list)
if __name__=='__main__':
    main()