import torch
import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from core.inference import get_max_preds
from tqdm import tqdm


class UnifiedSurgicalPipeline:
    def __init__(self, det_model, pose_model, model_type='vitpose', conf_threshold=0.5, device='cuda'):
        """
        model_type: 'vitpose' or 'hrnet'
        conf_threshold: Minimum YOLO confidence to process a tool
        """
        self.det_model = det_model 
        self.pose_model = pose_model
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device
        # Move model to device
        self.pose_model.to(self.device)
    
    def _process_hrnet_crop(self, img, bbox):
        # Add the scaling to match the training setup
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        scale = 1.1
        new_w, new_h = w * scale, h * scale
        
       
        px1 = int(cx - new_w / 2)
        py1 = int(cy - new_h / 2)
        px2 = int(cx + new_w / 2)
        py2 = int(cy + new_h / 2)
        
        # Clip to image bounds 
        H, W = img.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W - 1, px2), min(H - 1, py2)
        
        # Crop and convert color
        crop = img[py1:py2, px1:px2].copy()
        if crop.size == 0: return np.zeros((7, 3))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        h0, w0 = crop_rgb.shape[:2]
        input_img = cv2.resize(crop_rgb, (256, 192)) # input_w, input_h
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_tensor = (input_img.astype(np.float32) / 255.0 - mean) / std
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device).float()

        with torch.no_grad():
            output = self.pose_model(input_tensor)
            preds_local, maxvals = get_max_preds(output.cpu().numpy())
            
            hm_h, hm_w = 48, 64
            
            # Scale back to the PADDED crop size
            preds_global = np.zeros_like(preds_local)
            preds_global[0, :, 0] = (preds_local[0, :, 0] / hm_w) * w0 + px1
            preds_global[0, :, 1] = (preds_local[0, :, 1] / hm_h) * h0 + py1
            
            pred_kpts = np.concatenate([preds_global[0], maxvals[0]], axis=1)
            
        return pred_kpts


    def predict(self, img_path):
        img = cv2.imread(img_path)
        # YOLO Detection, consider only detections with confidence higher than the chosen conf_threshold
        det_results = self.det_model.predict(img, verbose=False, conf=self.conf_threshold)[0]
        
        # Get bboxes 
        bboxes = det_results.boxes.xyxy.cpu().numpy()
      
        
        if len(bboxes) == 0:
            return []

        if self.model_type == 'vitpose':
            # Prediction using Vitpose 
            return inference_topdown(self.pose_model, img, bboxes=bboxes)
        else:
            # Prediction unsing HRNet
            results = []
            for box in bboxes:
                results.append(self._process_hrnet_crop(img, box))
            return results