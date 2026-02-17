# a wrapper that would unify the output of inference of each model or pipeline
import torch
import cv2
import numpy as np
from mmpose.apis import inference_topdown
from core.inference import get_max_preds
from tqdm import tqdm

class UnifiedSurgicalPipeline:
    def __init__(self, det_model, pose_model, pose_model_type='vitpose', eval_setup='top_down', conf_threshold=0.5, device='cuda'):
        """
        model_type: 'vitpose' or 'hrnet' or 'yolo'
        eval_setup: 'top_down', 'full_frame', 'yolo_pose', 'kpt_only'
        conf_threshold: Minimum YOLO confidence to process a tool
        """
        self.det_model = det_model 
        self.pose_model = pose_model
        self.pose_model_type = pose_model_type
        self.conf_threshold = conf_threshold
        self.device = device
        self.eval_setup = eval_setup
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


    def predict(self, img_info):
        assert (self.eval_setup.lower() == 'top_down' and not(self.det_model== None)) , "ERROR: top_down pipeline needs a detection model definition"
        if self.eval_setup.lower() == 'top_down':
            return self._predict_top_down(img_info)
        if self.eval_setup.lower() == 'kpt_only':
            return self._predict_kpt_only(img_info)
        if self.eval_setup.lower() == 'yolo_pose':
            return self._predict_yolo_pose(img_info)
        if self.eval_setup.lower() == 'full_frame':
            return self._predict_full_frame(img_info)
        
    def _predict_top_down(self, img_path):
        """Implementation for VitPose and HRNet following YOLO detection"""
        img = cv2.imread(img_path)
        det_results = self.det_model.predict(img, verbose=False, conf=self.conf_threshold)[0]
        bboxes = det_results.boxes.xyxy.cpu().numpy()
        
        if len(bboxes) == 0:
            return []

        if self.pose_model_type == 'vitpose':
            # mmpose handles the crops internally
            mm_results = inference_topdown(self.pose_model, img, bboxes=bboxes)
            results = []
            for res in mm_results:
                kpts = res.pred_instances.keypoints[0]
                confs = res.pred_instances.keypoint_scores[0]
                results.append(np.concatenate([kpts, confs[:, None]], axis=1))
            return results
        else:
            # HRNet with your 1.1x manual crop scaling
            results = []
            for box in bboxes:
                results.append(self._process_hrnet_crop(img, box))
            return results


    def _predict_yolo_pose(self, img_path):
        """Native YOLOv8-pose inference (Returns [N, 7, 3])"""
        img = cv2.imread(img_path)
        # YOLOv8-pose models directly provide global coordinates
        results = self.pose_model.predict(img, verbose=False, conf=self.conf_threshold)[0]
        
        if results.keypoints is None:
            return []
        
        # results.keypoints.data is [N, 7, 3] (x, y, conf)
        return results.keypoints.data.cpu().numpy()

    def _predict_kpt_only(self, img_info):
        """
        Used for evaluating pose models on GT crops (standard pose test).
        img_info here is a dict from the dataset: {'img': tensor, 'bbox': [x1,y1,x2,y2]}
        """
        img = img_info['img_raw'] # Original BGR image
        bbox = img_info['bbox']   # [x1, y1, x2, y2]
        
        if self.pose_model_type == 'vitpose':
            # mmpose inference_topdown expects a list of bboxes
            # result is a list of PoseDataSample
            results = inference_topdown(self.pose_model, img, bboxes=np.array([bbox]))
            # Standardize to [7, 3]
            pred_kpts = results[0].pred_instances.keypoints[0]
            scores = results[0].pred_instances.keypoint_scores[0]
            return [np.concatenate([pred_kpts, scores[:, None]], axis=1)]
        else:
            # Standard HRNet crop processing using your 1.1x scaling logic
            return [self._process_hrnet_crop(img, bbox)]

    def _predict_full_frame(self, img_path):
        """
        Evaluates models that take the whole image as input (e.g., HRNet trained on full frame).
        """
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Standard input resize for your HRNet full-frame setup
        input_img = cv2.resize(img_rgb, (256, 192))
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_tensor = (input_img.astype(np.float32) / 255.0 - mean) / std
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            output = self.pose_model(input_tensor)
            preds_local, maxvals = get_max_preds(output.cpu().numpy())
            
            # heatmap size: 48x64
            hm_h, hm_w = 48, 64
            
            # Scale back directly to the FULL image size (W, H)
            preds_global = np.zeros_like(preds_local)
            preds_global[0, :, 0] = (preds_local[0, :, 0] / hm_w) * W
            preds_global[0, :, 1] = (preds_local[0, :, 1] / hm_h) * H
            
            # For full frame, we might have multiple instances in one output
            # (assuming output channels = 7 * num_instances)
            results = []
            num_instances = output.shape[1] // 7
            for i in range(num_instances):
                start = i * 7
                end = (i + 1) * 7
                inst_kpts = np.concatenate([preds_global[0, start:end, :], maxvals[0, start:end, :]], axis=1)
                results.append(inst_kpts)
                
        return results

    

    