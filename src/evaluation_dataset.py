import os
import yaml
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SurgPoseDatasetOneInstanceInference(Dataset):
    def __init__(self, img_root, ann_root, input_size=(256, 192), heatmap_size=(64, 48),
                 num_joints=7, sigma=2, transform=None, video_list=None):

        self.img_root = img_root
        self.ann_root = ann_root
        #self.input_size = input_size
        self.input_w = input_size[0]
        self.input_h = input_size[1]
        #self.heatmap_size = heatmap_size
        self.heatmap_w = heatmap_size[0]  # 64
        self.heatmap_h = heatmap_size[1]  # 48
        self.num_joints = num_joints
        self.sigma = sigma
        num_total = 0
        num_skipped = 0
        
        # test id there is a given video list (generated from the split file)
        if video_list is None:
            video_ids=  sorted(os.listdir(img_root))
        else:
            video_ids=  sorted([vid for vid in video_list if os.path.isdir(os.path.join(img_root, vid))])

        # get images and keypoints and bboxes
        self.samples = []
        for video_id in video_ids:
            img_dir = os.path.join(img_root, video_id)
            kp_dir = os.path.join(ann_root, video_id)
            if not os.path.isdir(img_dir):
                continue
            for img_name in sorted(os.listdir(img_dir)):
                if not img_name.endswith(".jpg"):
                    continue
                ann_path = os.path.join(kp_dir, img_name.replace(".jpg", ".yaml"))
                if not os.path.exists(ann_path):
                    continue

                with open(ann_path, "r") as f:
                    ann = yaml.safe_load(f)
                
                for obj in ann["objects"]:
                    joints = np.array(obj["keypoints"], dtype=np.float32)
                    vis = np.array(obj["visibility"], dtype=np.float32).reshape(-1, 1)
                    bbox = obj["bbox"]

                    num_total+=1

                    if not self.is_valid_sample(joints, vis, bbox):
                        num_skipped+=1
                        continue 

                    self.samples.append({
                        "img_path": os.path.join(img_dir, img_name),
                        "bbox": bbox,
                        "keypoints": joints,
                        "visibility": vis,
                        "obj_id": obj["id"]
                    })
            
            
          
        self.transform = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.augmentation = transform
        print(f"[SurgPoseDataset] Kept {len(self.samples)} / {num_total} samples " , f"(skipped {num_skipped})")

     

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_raw = cv2.imread(sample["img_path"])
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

   
        orig_kpts = np.concatenate([sample["keypoints"], sample["visibility"]], axis=1)
        orig_bbox = np.array(sample["bbox"], dtype=np.float32) # [x1, y1, x2, y2]

       
        joints = sample["keypoints"].copy()
        vis = sample["visibility"].copy()
        x1, y1, x2, y2 = map(int, sample["bbox"])
        

        scale = 1.1
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = w * scale, h * scale
        
       
        crop_x1 = max(0, int(cx - new_w / 2))
        crop_y1 = max(0, int(cy - new_h / 2))
        crop_x2 = min(W - 1, int(cx + new_w / 2))
        crop_y2 = min(H - 1, int(cy + new_h / 2))

        img_crop = img_raw[crop_y1:crop_y2, crop_x1:crop_x2]
        h_crop, w_crop = img_crop.shape[:2]
        img_crop = Image.fromarray(img_crop)

      
        joints[:, 0] -= crop_x1
        joints[:, 1] -= crop_y1

        if self.augmentation is not None:
            aug = self.augmentation(image=np.array(img_crop), keypoints=joints)
            img_tensor = aug['image']
            joints = np.array(aug['keypoints'], dtype=np.float32)
        else:
            img_tensor = self.transform(img_crop)
          
            joints[:, 0] = joints[:, 0] / w_crop * self.input_w
            joints[:, 1] = joints[:, 1] / h_crop * self.input_h

      
        mask = ((joints[:,0] < 0) | (joints[:,0] >= self.input_w) | 
                (joints[:,1] >= self.input_h) | (joints[:,1] < 0))
        vis[mask] = 0

    
        joints_hm = joints.copy()
        joints_hm[:, 0] = joints_hm[:, 0] / self.input_w * self.heatmap_w
        joints_hm[:, 1] = joints_hm[:, 1] / self.input_h * self.heatmap_h
        target, target_weight = self.generate_target(joints_hm, vis)

      
        return  img_tensor,torch.from_numpy(target),torch.from_numpy(target_weight),torch.from_numpy(orig_kpts),torch.from_numpy(orig_bbox),torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]),sample["obj_id"]
    
        
        
    def generate_target(self, joints, joints_vis):
        num_joints = self.num_joints
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:] = joints_vis
        target = np.zeros((num_joints, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        #target = np.zeros((num_joints, self.heatmap_w, self.heatmap_h), dtype=np.float32), probably this is more correct

        tmp_size = self.sigma * 3
        for joint_id in range(num_joints):
            mu_x = int(joints[joint_id][0])
            mu_y = int(joints[joint_id][1])

            if joints_vis[joint_id][0] == 0:
                continue
            
            #compute window to place the gaussian
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] #upper-left
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] #bottom-right
            #Check if the window is inside the heatmap
            if (ul[0] >= self.heatmap_w or ul[1] >= self.heatmap_h or
                br[0] < 0 or br[1] < 0):
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_h) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_w)
            img_y = max(0, ul[1]), min(br[1], self.heatmap_h)

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

 

  
    
    
    def is_valid_sample(self, joints, vis, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1

        # 1. bbox size check
        if bw < 20 or bh < 20:
            return False

        # 2. bbox area check
        if bw * bh < 400:
            return False

        # 3. keypoints inside bbox
        for (x, y), v in zip(joints, vis):
            if v> 0:
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    return False

        return True


##### Yolo evaluation class #######

class YoloPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(640, 640)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        h0, w0 = img.shape[:2]

        # 2. Load Labels (.txt file)
        label_path = os.path.join(self.label_dir, self.img_files[idx].rsplit('.', 1)[0] + '.txt')
        gt_instances = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = np.fromstring(line, sep=' ')
                    # data format: [class, x_c, y_c, w, h, k1_x, k1_y, k1_v, ...]
                    gt_instances.append(data)
        
        gt_instances = np.array(gt_instances) if len(gt_instances) > 0 else np.zeros((0, 5 + 7*3))

        # 3. Rescale coordinates from normalized (0-1) to original pixels
        # We evaluate in pixels to be precise
        gt_bboxes = gt_instances[:, 1:5].copy()
        gt_bboxes[:, [0, 2]] *= w0  # x_center, width
        gt_bboxes[:, [1, 3]] *= h0  # y_center, height
        
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        bboxes_xyxy = np.zeros_like(gt_bboxes)
        bboxes_xyxy[:, 0] = gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2
        bboxes_xyxy[:, 1] = gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2
        bboxes_xyxy[:, 2] = gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2
        bboxes_xyxy[:, 3] = gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2

        gt_kpts = gt_instances[:, 5:].reshape(-1, 7, 3)
        gt_kpts[:, :, 0] *= w0
        gt_kpts[:, :, 1] *= h0

        # 4. Prepare for YOLO inference
        # Resize image for the model but keep original for coordinate matching
        img_resized = cv2.resize(img, self.img_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() # [3, H, W]

        return { "img_path": img_path,
            "img": img_tensor,           # To be fed to YOLO
            "orig_img": img,             # For visualization if needed
            "gt_kpts": torch.as_tensor(gt_kpts),
            "gt_bboxes": torch.as_tensor(bboxes_xyxy),
            "h_w_orig": (h0, w0)
        }
############## HRNET FULL IMAGE evaluation dataset class #####################

class SurgPoseInferenceDataset(Dataset):
    def __init__(self, img_root, ann_root, input_size=(256, 192), heatmap_size=(64, 48),
                 num_joints=14, video_list=None):
        self.img_root = img_root
        self.ann_root = ann_root
        self.input_w, self.input_h = input_size
        self.heatmap_w, self.heatmap_h = heatmap_size
        self.num_joints = num_joints
        
      
        if video_list is None:
            video_ids = sorted(os.listdir(img_root))
        else:
            video_ids = sorted([vid for vid in video_list if os.path.isdir(os.path.join(img_root, vid))])

        self.samples = []
        for video_id in video_ids:
            img_dir = os.path.join(img_root, video_id)
            if not os.path.isdir(img_dir): continue
            
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith(".jpg"):
                    self.samples.append(os.path.join(img_dir, img_name))

        self.transform = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]

        # Load image and keep original size for coordinate scaling
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        h0, w0 = img_raw.shape[:2]
        
        img_pil = Image.fromarray(img_raw)
        img_tensor = self.transform(img_pil)

        # Metadata needed for matching and OKS calculation
        meta = {
            "img_path": img_path,
            "h_w_orig": torch.tensor([h0, w0]),
        }

        return {
            "img": img_tensor,
            "meta": meta
        }