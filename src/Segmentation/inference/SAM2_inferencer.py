import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from ultralytics import SAM



import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import SAM
from tqdm import tqdm

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import SAM

class SAM2SegmentationInferencer:
    def __init__(self, model_path="sam2_b.pt", device="cuda", area_threshold=400):
        self.model = SAM(model_path)
        self.device = device
        self.area_threshold = area_threshold

    def run_inference(self, source_root, bbox_root, target_root, test_video_ids, scale=1.1):
        mask_out = Path(target_root) / "binary_masks"
        plot_out = Path(target_root) / "overlay_plots"
        mask_out.mkdir(parents=True, exist_ok=True)
        plot_out.mkdir(parents=True, exist_ok=True)

        for vid_id in test_video_ids:
            vid_img_path = os.path.join(source_root, vid_id)
            vid_bbox_path = os.path.join(bbox_root, vid_id)
            if not os.path.isdir(vid_img_path): continue

            frames = sorted([f for f in os.listdir(vid_img_path) if f.lower().endswith('.jpg')])
            
            for frame_name in tqdm(frames, desc=f"Video {vid_id}"):
                base_name = os.path.splitext(frame_name)[0]
                img_path = os.path.join(vid_img_path, frame_name)
                yaml_path = os.path.join(vid_bbox_path, f"{base_name}.yaml")

                if not os.path.exists(yaml_path): continue
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                img_bgr = cv2.imread(img_path)
                h, w = img_bgr.shape[:2]
                final_mask = np.zeros((h, w), dtype=np.uint8)
                
                # We collect all prompts for the overlay plotting
                all_frame_bboxes = []
                all_frame_points = []
                for obj in data.get('objects', []):
                    bbox = obj.get('bbox', None)
                    kpts = obj.get('keypoints', [])
                    
                    # Reset prompts for each object
                    curr_points = None
                    curr_labels = None
                    curr_bbox = None
                    bbox_check = False
                    
                    # Prepare bbox if it passes quality checks
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        
                        if w > 50 and h > 50 and w * h > 10000:
                            # Scaling the bbox to a alarger scale  
                            # Calculate center and new dimensions
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            new_w, new_h = w * scale, h * scale
                            nx1 = max(0, int(cx - new_w / 2))
                            ny1 = max(0, int(cy - new_h / 2))
                            nx2 = min(img_bgr.shape[1], int(cx + new_w / 2))
                            ny2 = min(img_bgr.shape[0], int(cy + new_h / 2))
                            
                            scaled_bbox = [nx1, ny1, nx2, ny2]
                            
                            #Use the scaled box for inference and overlay
                            curr_bbox = [scaled_bbox]
                            bbox_check = True
                            all_frame_bboxes.append(scaled_bbox)
                    
                    # Prepare points if they exist
                    if not bbox_check:
                        if len(kpts) > 0:
                            indices = [0, len(kpts)//2, len(kpts)-1]
                            # Wrap in an extra list so SAM treats them as points for ONE object
                            curr_points = [[kpts[i] for i in indices]] 
                            curr_labels = [[1, 1, 1]] 
                            all_frame_points.extend([kpts[i] for i in indices])

                
                    results = self.model.predict(
                        source=img_path,
                        bboxes=curr_bbox,
                        points=curr_points,
                        labels=curr_labels,
                        device=self.device,
                        verbose=False
                    )

                    if results[0].masks is not None:
                        for m in results[0].masks.data:
                            mask_np = (m.cpu().numpy() > 0).astype(np.uint8) * 255
                            final_mask = cv2.bitwise_or(final_mask, mask_np)

                # 5. Save the combined binary mask
                cv2.imwrite(str(mask_out / f"{base_name}.png"), final_mask)
                
                # 6. Save the visual overlay with all boxes/points shown
                self._save_diagnostic_overlay(
                    img_bgr=img_bgr, 
                    mask=final_mask, 
                    bboxes=all_frame_bboxes, 
                    points=all_frame_points, 
                    save_path=str(plot_out / f"{base_name}_overlay.jpg")
                )

    def _save_diagnostic_overlay(self, img_bgr, mask, bboxes, points, save_path):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        
        # Apply green mask color
        overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        
        plt.figure(figsize=(12, 7))
        plt.imshow(blended)
        ax = plt.gca()

        # Plot BBoxes (Red)
        for box in bboxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        # Plot Keypoints used (Yellow Stars)
        if points:
            pts = np.array(points)
            plt.scatter(pts[:, 0], pts[:, 1], color='yellow', marker='*', s=50, edgecolor='black')

        plt.axis('off')
        plt.title(f"SAM 2 Result (Green) | BBox (Red) | Kpts (Yellow)")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
# class SAM2SegmentationInferencer:

#     def __init__(self, model_path="sam2_b.pt", device="cuda"):
#         print(f"Initializing SAM 2 on {device}...")
#         self.model = SAM(model_path)
#         self.device = device

#     def run_inference(self, source_root, bbox_root, target_root, video_ids):
#         """
#         source_root: Path to image video folders (000033, etc.)
#         bbox_root: Path to YAML video folders
#         target_root: Where to save masks and overlay plots
#         """
#         mask_out = Path(target_root) / "binary_masks"
#         plot_out = Path(target_root) / "overlay_plots"
#         mask_out.mkdir(parents=True, exist_ok=True)
#         plot_out.mkdir(parents=True, exist_ok=True)

#         for vid_id in video_ids:
#             vid_img_path = os.path.join(source_root, vid_id)
#             vid_bbox_path = os.path.join(bbox_root, vid_id)

#             if not os.path.isdir(vid_img_path):
#                 continue

#             # Get frames and ensure they match annotations
#             frames = sorted([f for f in os.listdir(vid_img_path) if f.lower().endswith('.jpg')])
            
#             for frame_name in tqdm(frames, desc=f"Video {vid_id}"):
#                 base_name = os.path.splitext(frame_name)[0]
#                 img_path = os.path.join(vid_img_path, frame_name)
#                 yaml_path = os.path.join(vid_bbox_path, f"{base_name}.yaml")

#                 if not os.path.exists(yaml_path):
#                     continue

#                 #Load Image and YAML
#                 img_bgr = cv2.imread(img_path)
#                 with open(yaml_path, 'r') as f:
#                     data = yaml.safe_load(f)

#                 #Extract BBoxes
#                 bboxes = []
#                 if 'objects' in data:
#                     for obj in data['objects']:
#                         if obj.get('bbox'):
#                             bboxes.append(obj['bbox'])

#                 if not bboxes:
#                     continue

#                 #SAM 2 Inference using bboxes as prompts
#                 results = self.model.predict(
#                     source=img_path,
#                     bboxes=bboxes,
#                     device=self.device,
#                     verbose=False
#                 )

#                 #Create combined binary mask
#                 h, w = img_bgr.shape[:2]
#                 final_mask = np.zeros((h, w), dtype=np.uint8)
                
#                 if results[0].masks is not None:
#                     for m in results[0].masks.data:
#                         mask_np = (m.cpu().numpy() > 0).astype(np.uint8) * 255
#                         final_mask = cv2.bitwise_or(final_mask, mask_np)

#                 #Save Results
#                 save_name = f"{base_name}.png"
#                 cv2.imwrite(str(mask_out / save_name), final_mask)
                
#                 #Save Diagnostic Overlay
#                 self._save_overlay(img_bgr, final_mask, bboxes, plot_out / save_name)

#     def _save_overlay(self, img_bgr, mask, bboxes, save_path):
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
#         # Create green semi-transparent mask
#         overlay = img_rgb.copy()
#         overlay[mask > 0] = [0, 255, 0]
#         blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        
#         plt.figure(figsize=(10, 6))
#         plt.imshow(blended)
#         ax = plt.gca()
        
#         # Draw the GT BBoxes in Red
#         for box in bboxes:
#             x1, y1, x2, y2 = box
#             rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
#             ax.add_patch(rect)
            
#         plt.axis('off')
#         plt.title("SAM 2 Mask (Green) guided by GT BBox (Red)")
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         plt.close()


