import os
import yaml
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SurgPoseDataset(Dataset):
    def __init__(self, img_root, ann_root, input_size=(256, 192), heatmap_size=(64, 48),
                 num_joints=14, sigma=2, transform=None, video_list=None, keep_kpt_gt= False):

        self.img_root = img_root
        self.ann_root = ann_root
        # self.input_size = input_size
        self.input_w = input_size[0] # 256
        self.input_h = input_size[1] # 192
        # self.heatmap_size = heatmap_size
        self.heatmap_w = heatmap_size[0]  # 64
        self.heatmap_h = heatmap_size[1]  # 48
        self.num_joints = num_joints
        self.sigma = sigma
        self.keep_kpt_gt = keep_kpt_gt #added
        
        # test id there is a given video list (generated from the split file)
        if video_list is None:
            video_ids=  sorted(os.listdir(img_root))
        else:
            video_ids=  sorted([vid for vid in video_list if os.path.isdir(os.path.join(img_root, vid))])

        # get images and keypoints
        self.samples = []
        for video_id in video_ids:
            img_dir = os.path.join(img_root, video_id)
            kp_dir = os.path.join(ann_root, video_id)
            if not os.path.isdir(img_dir):
                continue
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith(".jpg"):
                    base = os.path.splitext(img_name)[0]
                    ann_path = os.path.join(kp_dir, f"{base}.yaml")
                    if os.path.exists(ann_path):
                        self.samples.append((os.path.join(img_dir, img_name), ann_path))

      

        self.transform = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.augmentation = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        
        with open(ann_path, "r") as f:
            ann = yaml.safe_load(f)
      
        joints = np.array(ann["keypoints"], dtype=np.float32)
       
        vis = np.array(ann["visibility"], dtype=np.float32).reshape(self.num_joints, 1)
        if self.keep_kpt_gt:
            joints_gt = joints.copy() # added
       
        #apply transformations
        if self.augmentation :
            h0, w0 = img.size[1], img.size[0]
            aug = self.augmentation(image=np.array(img), keypoints=joints)
            img = aug['image']
            joints = np.array(aug['keypoints'], dtype=np.float32)
   
        else:
            h0, w0 = img.size[1], img.size[0]
            img = self.transform(img)
            joints[:, 0] = joints[:, 0] / w0 * self.input_w
            joints[:, 1] = joints[:, 1] / h0 * self.input_h
          
         
   
        mask = ((joints[:,0]< 0) | (joints[:,0]>= self.input_w) | (joints[:,1] >= self.input_h)| (joints[:,1] < 0 ))
        vis[mask]=0

        #Scale joints into heatmap resolution
        joints_hm = joints.copy()
       
        joints_hm[:, 0] = joints_hm[:, 0] / self.input_w * self.heatmap_w
        joints_hm[:, 1] = joints_hm[:, 1] / self.input_h * self.heatmap_h
        
        # Create heatmaps from the keypoints coordinates and the visibility mask
        target, target_weight = self.generate_target(joints_hm, vis)
        
        if self.keep_kpt_gt:
            return img, torch.from_numpy(target), torch.from_numpy(target_weight), torch.from_numpy(joints_gt) # added
        else:
            return img, torch.from_numpy(target), torch.from_numpy(target_weight), None

    def generate_target(self, joints, joints_vis):
        num_joints = self.num_joints
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:] = joints_vis
        target = np.zeros((num_joints, self.heatmap_h, self.heatmap_w), dtype=np.float32)
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

class SurgPoseDatasetOneInstance(Dataset):
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

        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        H, W = img.shape[:2]
        joints = sample["keypoints"].copy()
        vis = sample["visibility"].copy()
        x1, y1, x2, y2 = map(int, sample["bbox"])

        # pad bounding box to avoid loosing keypoints
        
        scale=1.1
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = w * scale
        new_h = h * scale

        x1 = int(cx - new_w / 2)
        x2 = int(cx + new_w / 2)
        y1 = int(cy - new_h / 2)
        y2 = int(cy + new_h / 2)

        #Clip to the image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for {sample['img_path']}")

        img = img[y1:y2, x1:x2]
        
        img = Image.fromarray(img)

       
       
        joints[:, 0] -= x1
        joints[:, 1] -= y1
       
        #apply transformations
        if self.augmentation is not None:
            h0, w0 = img.size[1], img.size[0]
            aug = self.augmentation(image=np.array(img), keypoints=joints)
            img = aug['image']
            joints = np.array(aug['keypoints'], dtype=np.float32)
        
   
        else:
            h0, w0 = img.size[1], img.size[0]
            img = self.transform(img)
            joints[:, 0] = joints[:, 0] / w0 * self.input_w
            joints[:, 1] = joints[:, 1] / h0 * self.input_h
         
   
        mask = ((joints[:,0]< 0) | (joints[:,0]>= self.input_w) | (joints[:,1] >= self.input_h)| (joints[:,1] < 0 ))
     
        vis[mask]=0

        #Scale joints into heatmap resolution
        joints_hm = joints.copy()
        joints_hm[:, 0] = joints_hm[:, 0] / self.input_w * self.heatmap_w
        joints_hm[:, 1] = joints_hm[:, 1] / self.input_h * self.heatmap_h

        #Accounting for cases where num_joint=14
        
        if self.num_joints> len(joints_hm): # added
         
            new_joints = np.zeros((self.num_joints, 2), dtype=np.float32)
            new_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
            new_joints[:len(joints_hm)] = joints_hm
            new_vis[:len(vis)] = vis
            joints_hm = new_joints
            vis = new_vis
      
        # Create heatmaps from the keypoints coordinates and the visibility mask
        target, target_weight = self.generate_target(joints_hm, vis)

        return img, torch.from_numpy(target), torch.from_numpy(target_weight), sample["obj_id"]

    def generate_target(self, joints, joints_vis):
        num_joints = self.num_joints
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:] = joints_vis
        target = np.zeros((num_joints, self.heatmap_h, self.heatmap_w), dtype=np.float32)

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




