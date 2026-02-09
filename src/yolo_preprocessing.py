import json
import cv2
from pathlib import Path
import os
import yaml
import re
import zipfile
from tqdm import tqdm


############## YOLO FOR OBJECT DETECTION ############################################


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn



def build_yolo_dataset(split_file, dataset_root, surgpose_path, zip_folder, class_id=0):
    #use the split file to rearrange images in train val and test files
    # access zip to get open bbox files 
    # convert bbox and save them in accurate location
    
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root, exist_ok = True)
    
    images_path=os.path.join(dataset_root,'images')
    labels_path = os.path.join(dataset_root, 'labels')
    
    os.makedirs(images_path, exist_ok= True)
    os.makedirs(labels_path, exist_ok= True)

    subfolders=['train', 'val','test']
    # class_map = {"obj1": 0, "obj2": 1}

 
    for stage in tqdm(subfolders, desc="Processing stages", unit="stage"):
        
        stage_img_path = os.path.join(images_path,stage)
        stage_lbl_path = os.path.join(labels_path, stage)
        
        os.makedirs(stage_img_path, exist_ok= True)
        os.makedirs(stage_lbl_path, exist_ok= True)

        with open(split_file, "r") as f:
            splits = yaml.safe_load(f)
        stage_video_list = splits[stage]  
        for vid in tqdm(stage_video_list, desc=f'Processing videos in {stage}', unit='video'):
            zip_path= os.path.join(zip_folder, vid+'.zip')
            current= os.path.join(surgpose_path, vid)
            frames= os.listdir(current)
            for frame in frames:
                
                frame_path = os.path.join(current, frame)
                img = cv2.imread(frame_path)
                h, w = img.shape[:2]
                camera= 'right'
                if 'left' in frame:
                    camera='left'
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    bbox_files = [f for f in zip_ref.namelist() if f.lower().endswith(".json") and camera in f.lower()]
                    if not bbox_files:
                        continue
                    bbox_path = bbox_files[0]
                    
                    with zip_ref.open(bbox_path) as f:
                        bboxes = json.load(f)
                
                m = re.search(r'frame_(\d+)', frame)
                if m is None:
                    continue
                frame_id = str(int(m.group(1)))
               
                
                if frame_id not in bboxes:
                    continue

                objs= bboxes[frame_id]

                label_lines= []
                for obj_name, bbox in objs.items():
                    xc, yc, bw, bh = coco_to_yolo_bbox(bbox, w, h)
                    
                    # class_id = class_map[obj_name]
                    label_lines.append(
                        f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    )
              

                # Save image
                out_img_path = os.path.join(stage_img_path,frame)
                cv2.imwrite(str(out_img_path), img)

                # Save label
                label_path = os.path.join(stage_lbl_path,frame.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

############ YOLO for POSE ESTIMATION ################################


def write_pose_line(
    class_id,
    bbox_xywh,
    keypoints,
    visibility,
    img_w,
    img_h
):
    xc, yc, bw, bh = coco_to_yolo_bbox(bbox_xywh, img_w, img_h)
    line = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    for (x, y), v in zip(keypoints, visibility):
        if v == 0:
            line += " 0 0 0"
        else:
            line += f" {x/img_w:.6f} {y/img_h:.6f} {int(v)}"

    return line

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def parse_bboxes_for_frame_xywh(frame_bboxes):
    """
    Input: {"obj1": [x,y,w,h], "obj2": [...]}
    Output: {"left": bbox_xywh, "right": bbox_xywh}
    """
    boxes = []
    for bbox in frame_bboxes.values():
        x1, y1, x2, y2 = xywh_to_xyxy(bbox)
        cx = (x1 + x2) / 2.0
        boxes.append((cx, bbox))

    boxes = sorted(boxes, key=lambda x: x[0])

    return {
        "left": boxes[0][1],
        "right": boxes[1][1]
    }

def build_yolo_pose_dataset(split_file,dataset_root,frames_root,keypoints_root,zip_root,class_id=0):
    labels_root = os.path.join(dataset_root, "labels_pose")
    images_root = os.path.join(dataset_root, "images")

    os.makedirs(labels_root, exist_ok=True)

    with open(split_file, "r") as f:
        splits = yaml.safe_load(f)

    for stage in ["train", "val", "test"]:
        print(f"\nProcessing {stage} split")

        stage_img_dir = os.path.join(images_root, stage)
        stage_lbl_dir = os.path.join(labels_root, stage)

        os.makedirs(stage_lbl_dir, exist_ok=True)

        for vid in tqdm(splits[stage], desc=f"{stage} videos"):
            zip_path = os.path.join(zip_root, vid + ".zip")
            frame_dir = os.path.join(frames_root, vid)
            kp_dir = os.path.join(keypoints_root, vid)

            if not os.path.exists(zip_path):
                continue

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                bbox_jsons = {}
                for cam in ["left", "right"]:
                    files = [f for f in zip_ref.namelist()
                             if f.lower().endswith(".json") and cam in f.lower()]
                    if files:
                        with zip_ref.open(files[0]) as f:
                            bbox_jsons[cam] = json.load(f)

                for frame in os.listdir(frame_dir):
                    if not frame.endswith(".jpg"):
                        continue

                    camera = "left" if "left" in frame else "right"
                    if camera not in bbox_jsons:
                        continue

                    m = re.search(r'frame_(\d+)', frame)
                    if m is None:
                        continue

                    frame_id = str(int(m.group(1)))
                    if frame_id not in bbox_jsons[camera]:
                        continue

                    img_path = os.path.join(frame_dir, frame)
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]

                    kp_yaml = os.path.join(kp_dir, frame.replace(".jpg", ".yaml"))
                    if not os.path.exists(kp_yaml):
                        continue

                    with open(kp_yaml, "r") as f:
                        ann = yaml.safe_load(f)

                    keypoints = ann["keypoints"]
                    visibility = ann["visibility"]

                    bbox_per_inst = parse_bboxes_for_frame_xywh(
                        bbox_jsons[camera][frame_id]
                    )

                    label_lines = []

                    for i, inst in enumerate(["right", "left"]):
                        kp = keypoints[i*7:(i+1)*7]
                        vis = visibility[i*7:(i+1)*7]
                        bbox = bbox_per_inst[inst]

                        label_lines.append(
                            write_pose_line(
                                class_id, bbox, kp, vis, w, h
                            )
                        )

                    out_lbl = os.path.join(
                        stage_lbl_dir,
                        frame.replace(".jpg", ".txt")
                    )
                    with open(out_lbl, "w") as f:
                        f.write("\n".join(label_lines))



############### ONE INSTANCE HRNET DATASET MODIFICATIONS  ############

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def parse_bboxes_for_frame(frame_bboxes):
    """
    frame_bboxes: dict like {"obj1": [...], "obj2": [...]}
    returns: {"left": bbox_xyxy, "right": bbox_xyxy}
    """

    boxes = []
    for obj_id, bbox in frame_bboxes.items():
        x1, y1, x2, y2 = xywh_to_xyxy(bbox)
        cx = (x1 + x2) / 2.0
        boxes.append((cx, [x1, y1, x2, y2]))

    # sort by x center
    boxes = sorted(boxes, key=lambda x: x[0])

    return {
        "left": boxes[0][1],
        "right": boxes[1][1]
    }

def convert_yaml(old_yaml_path, new_yaml_path, bbox_per_instance):
    with open(old_yaml_path, "r") as f:
        ann = yaml.safe_load(f)

    keypoints = ann["keypoints"]
    visibility = ann["visibility"]

    assert len(keypoints) == 14, "Expected 14 keypoints (2 instruments * 7)"

    objects = []

    for i, name in enumerate(["right", "left"]):
        kp = keypoints[i*7:(i+1)*7]
        vis = visibility[i*7:(i+1)*7]

        obj = {
            "id": name,
            "keypoints": kp,
            "visibility": vis,
            "bbox": bbox_per_instance[name]
        }
        objects.append(obj)

    new_ann = {
        "frame_id": ann["frame_id"],
        "video_id": ann["video_id"],
        "objects": objects
    }
   
    
    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(new_ann, f, sort_keys=False)



def add_bboxes_to_surgpose_data(frame_root, kpt_root, zip_root, save_ann):
    videos = os.listdir(frame_root)

    for vid in tqdm(videos , desc='processing videos', unit='video'):
        zip_path = os.path.join(zip_root, vid + ".zip")
        vid_dir = os.path.join(frame_root, vid)
   
        kpt_dir = os.path.join(kpt_root, vid)
        new_kpt_dir = os.path.join(save_ann, vid)
        os.makedirs(new_kpt_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
        
            bbox_jsons = {}
            for cam in ["left", "right"]:
                files = [f for f in zip_ref.namelist()
                         if f.lower().endswith(".json") and cam in f.lower()]
                if files:
                    with zip_ref.open(files[0]) as f:
                        bbox_jsons[cam] = json.load(f)

            for frame in os.listdir(vid_dir):
          
                if not frame.endswith(".jpg"):
                    continue

                camera = "left" if "left" in frame else "right"
                if camera not in bbox_jsons:
                    continue

                m = re.search(r'frame_(\d+)', frame)
                if m is None:
                    continue

                frame_id = str(int(m.group(1)))
            
                if frame_id not in bbox_jsons[camera]:
                    continue

                bbox_dict = parse_bboxes_for_frame(
                    bbox_jsons[camera][frame_id]
                )

                old_yaml = os.path.join(kpt_dir, frame.replace(".jpg", ".yaml"))
                new_yaml = os.path.join(new_kpt_dir, frame.replace(".jpg", ".yaml"))
               # print(old_yaml)
                if not os.path.exists(old_yaml):
                    continue

                convert_yaml(old_yaml, new_yaml, bbox_dict)
       







    

if __name__ == '__main__':
    # split_path='/srv/homes/onbo10/thesis_Ons/MiniSurgPose/Extracted3_left_right/video_split.yaml'
    # dataset_root='/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/yolo_formated_surgpose_2'
    # surgpose_path= '/srv/homes/onbo10/thesis_Ons/MiniSurgPose/Extracted3_left_right/extracted_frames'
    # zipfolder= '/srv/homes/onbo10/thesis_Ons/MiniSurgPose'
    # build_yolo_dataset(split_path,dataset_root, surgpose_path, zipfolder)
    
    ########## HRNET one instance ######
    # frame_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_frames'
    
    # kpt_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_keypoints'
    # zip_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData'
    # save_ann= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_bboxes_kpts'
    # add_bboxes_to_surgpose_data(frame_root,kpt_root,zip_root,save_ann)

    ############# YOLO POSE ESTIMATION ###################
    split_path='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/video_split.yaml'
    dataset_root='/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/yolo_formated_surgpose'
    frames_path= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_frames'
    keypoints_path='/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_keypoints'
    zipfolder= '/srv/homes/onbo10/thesis_Ons/SurgePoseData'
    build_yolo_pose_dataset(split_path,dataset_root, frames_path, keypoints_path, zipfolder)