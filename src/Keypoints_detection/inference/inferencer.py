import cv2
import torch
import numpy as np
from core.inference import get_max_preds

class keypointsDetectionInferencer:
    def __init__(self, model, model_type, device='cuda',input_size= (256, 192), heatmap_size= (64, 48)):
        """model_type:'yolopose', 'hrnet' or 'pipline' """
        self.model= model
        self.device = device
        self.model_type= model_type.lower()
        if self.model_type == 'hrnet':
            self.input_size= input_size
            self.heatmap_size= heatmap_size

        
    def predict(self, img_path):
        """Returns keypoints in a unified (N, 7, 2) numpy format"""
        if self.model_type == 'yolopose':
            results = self.model(img_path, verbose=False)
            # YOLO returns [Num_Tools, Num_Kpts, 2]
            return results[0].keypoints.xy.cpu().numpy()
        elif self.model_type == 'pipline':
            pipeline_results = self.model.predict(img_path) # returns a list
            
            final_coords = []
            for res in pipeline_results:
                if self.model.pose_model_type == 'vitpose':
                    # Extract kpts from MMPose DataSample
                    kpts = res.pred_instances.keypoints[0] # (7, 2)
                    final_coords.append(kpts)
                elif self.model.pose_model_type == 'hrnet':
                    # Extract (x, y) from (x, y, conf)
                    kpts = res[:, :2] # (7, 2)
                    final_coords.append(kpts)
            return np.array(final_coords)
        
        elif self.model_type == 'hrnet':
            results = self._run_hrnet_inference(img_path, self.input_size, self.heatmap_size)
            return results

 
    def _run_hrnet_inference(self, img_path, input_size, heatmap_size):
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        input_img = cv2.resize(img_rgb, input_size) # input_w, input_h
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_tensor = (input_img.astype(np.float32) / 255.0 - mean) / std
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device).float()

        with torch.no_grad():
            output = self.model(input_tensor)
            preds_local, _ = get_max_preds(output.cpu().numpy())
            
            hm_w, hm_h = heatmap_size

            preds_global = np.zeros_like(preds_local)
            preds_global[0, :, 0] = (preds_local[0, :, 0] / hm_w) * w0
            preds_global[0, :, 1] = (preds_local[0, :, 1] / hm_h) * h0
            
            pred_kpts = preds_global[0]
            
        return pred_kpts
