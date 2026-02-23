import os
import yaml
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HRNetEvaluationDataset(Dataset):
    def __init__(self, img_root, ann_root, input_size=(256, 192), 
                 mode='cropped', video_list=None):
        """
        Args:
            mode: 'cropped' (returns one instance per sample) or 
                  'full' (returns full image for pipeline evaluation)
        """
        self.img_root = img_root
        self.ann_root = ann_root
        self.input_w, self.input_h = input_size
        self.mode = mode
        
        # Determine which videos to include
        if video_list is None:
            video_ids = sorted(os.listdir(img_root))
        else:
            video_ids = sorted([vid for vid in video_list if os.path.isdir(os.path.join(img_root, vid))])

        self.samples = []
        self._load_samples(video_ids)

        self.transform = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self, video_ids):
        for video_id in video_ids:
            img_dir = os.path.join(self.img_root, video_id)
            ann_dir = os.path.join(self.ann_root, video_id)
            if not os.path.isdir(img_dir): 
                continue

            for img_name in sorted(os.listdir(img_dir)):
                if not img_name.endswith(".jpg"): 
                    continue
                
                img_path = os.path.join(img_dir, img_name)
                ann_path = os.path.join(ann_dir, img_name.replace(".jpg", ".yaml"))
                
                if self.mode == 'full':
                    # In 'full' mode, one sample = one image path
                    self.samples.append({"img_path": img_path, "ann_path": ann_path})
                else:
                    # In 'cropped' mode, one sample = one tool instance
                    if not os.path.exists(ann_path): continue
                    with open(ann_path, "r") as f:
                        ann = yaml.safe_load(f)
                    for obj in ann["objects"]:
                        if self.is_valid_sample(obj["keypoints"], obj["visibility"], obj["bbox"]):
                            self.samples.append({
                                "img_path": img_path,
                                "bbox": obj["bbox"],
                                "keypoints": np.array(obj["keypoints"], dtype=np.float32),
                                "visibility": np.array(obj["visibility"], dtype=np.float32).reshape(-1, 1),
                                "obj_id": obj["id"]
                            })

    def is_valid_sample(self, joints, vis, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1

        # bbox size check
        if bw < 20 or bh < 20:
            return False

        # bbox area check
        if bw * bh < 400:
            return False

        # keypoints inside bbox
        for (x, y), v in zip(joints, vis):
            if v> 0:
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    return False

        return True


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_raw = cv2.imread(s["img_path"])
        orig_img = img_raw.copy()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        H, W = img_raw.shape[:2]

        if self.mode == 'full':
            # Returns full image and path for the Unified Pipeline
            img_pil = Image.fromarray(img_raw)
            return {
                "img": self.transform(img_pil),
                "orig_img": orig_img, 
                "img_path": s["img_path"],
                "h_w_orig": torch.tensor([H, W])
            }

        else:
            # Returns cropped instance for HRNet/ViTPose GT evaluation
            gt_kpts = np.concatenate([s["keypoints"], np.array(s["visibility"])], axis=1)
            orig_bbox = np.array(s["bbox"], dtype=np.float32)
            
            x1, y1, x2, y2 = map(int, s["bbox"])
            scale = 1.1
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            new_w, new_h = w * scale, h * scale
            
        
            crop_x1 = max(0, int(cx - new_w / 2))
            crop_y1 = max(0, int(cy - new_h / 2))
            crop_x2 = min(W - 1, int(cx + new_w / 2))
            crop_y2 = min(H - 1, int(cy + new_h / 2))

            img_crop = img_raw[crop_y1:crop_y2, crop_x1:crop_x2]
            img_crop = Image.fromarray(img_crop)
            img_tensor = self.transform(img_crop)
           
            
            return {
                "img": img_tensor, # fed to network
                "orig_img": orig_img, # for visualization if needed
                "gt_kpts": torch.from_numpy(gt_kpts),
                "gt_bbox": orig_bbox,
                "crop_coords": torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]),
                "img_path": s["img_path"]
            }

class YoloPoseEvaluationDataset(Dataset):
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