import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.Keypoints_detection.evaluation.evaluation_utils import get_gt_from_hrnet_label_files
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio


class TrainingVisualizer:
    def __init__(self, log_path, save_dir="plots"):
        self.log_path = log_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.df = self._load_log()

    def _load_log(self):
        """Loads log file regardless (csv or json)"""
        if self.log_path.endswith('.csv'):
            df = pd.read_csv(self.log_path)
            #df.columns = [c.strip() for c in df.columns]
            return df
        elif self.log_path.endswith('.json'):
            data = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return pd.DataFrame(data)

    def plot_losses(self, train_col, val_col=None, title="Loss Curve", filename="loss.png"):
        plt.figure(figsize=(10, 5))
        
        # Plot training loss
        if train_col and train_col in self.df.columns:
            plt.plot(self.df[train_col], label=f'Train ({train_col})', color='blue')
        
        # Plot validation loss if it exists
        if val_col and val_col in self.df.columns:
            plt.plot(self.df[val_col], label=f'Val ({val_col})', color='red')
        
        plt.title(title)
        plt.xlabel('Epochs/Steps')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to: {save_path}")
    
    def plot_vitpose_accuracy(self, metric_col='coco/AP', title='Validation mAP', filename='accuracy.png'):
        """plots the validation metric mAP from ViTpose training"""
        # Extract only the rows that actually have the metric (remove NaNs)
        clean_df = self.df[['step', metric_col]].dropna()
        
        if clean_df.empty:
            print(f"Error: No data found for {metric_col}")
            return

        plt.figure(figsize=(10, 6))
        
        # Plot with markers so points are visible even if disconnected
        plt.plot(clean_df['step'], clean_df[metric_col], 
                marker='o', linestyle='-', color='red', 
                linewidth=2, markersize=6, label=metric_col)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('mAP Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()
       

class PoseVisualizer:

    def __init__(self, output_dir="visual_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Surgical tool keypoint colors (7 points)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                       (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def visualize_instance(self, img_path, gt_kpts, pred_kpts, name="result_0"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        
        # Plot GT in Green
        ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1], c='lime', marker='o', s=40, label='GT')
        # Plot Pred in Red
        ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], c='red', marker='x', s=40, label='Pred')
        
        # Draw lines between keypoints to show tool structure if needed
        ax.set_title(f"{name}")
        ax.legend()
        plt.axis('off')
        
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"), bbox_inches='tight')
        plt.show()
    
    def run_inference_visualization(self,inferencer,inference_indices,img_paths,lbl_paths):
        """Runs inference on imagesof randomply determined indices and plots the results"""
        for idx in inference_indices:
            img_path = img_paths[idx]
            lbl_path = lbl_paths[idx]
            gt_kpts, vis = get_gt_from_hrnet_label_files(lbl_path)
            pred_kpts = inferencer.predict(img_path)
            pred_kpts = pred_kpts.reshape(-1,2)
            frame_name= os.path.basename(img_path).split('.')[0]
            self.visualize_instance(img_path,gt_kpts,pred_kpts, name = f'{frame_name}, index = {idx}')



class SegmentationVisualizer:
    def __init__(self, alpha=0.4, cmap_gt='spring', cmap_pred='autumn'):
        """
        visualizer for segmentation tasks.
        """
        self.alpha = alpha
        self.cmap_gt = cmap_gt
        self.cmap_pred = cmap_pred

    def _overlay_mask(self, ax, image, mask, cmap, title):
        """Internal helper to mask background and overlay on axis."""
        ax.imshow(image)
        if mask is not None:
            # Mask the background to make it transparent
            masked_data = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked_data, cmap=cmap, alpha=self.alpha)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def plot_comparison(self, image, gt_mask, pred_mask, title_base="Case", metrics=None, save_path=None, display=True):
        """
        Creates a 1x2 or 1x3 plot depending on provided masks.
        metrics: dict of {metric_name: value} to display in title.
        """
        num_plots = 2 if gt_mask is not None and pred_mask is not None else 1
        fig, axes = plt.subplots(1, num_plots, figsize=(9 * num_plots, 6))
        
        # If only one plot, axes isn't a list
        if num_plots == 1: axes = [axes]

        # Formatting metrics string
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]) if metrics else ""

        # GT Overlay
        if gt_mask is not None:
            self._overlay_mask(axes[0], image, gt_mask, self.cmap_gt, f"{title_base}\nGround Truth")

        # Pred Overlay
        if pred_mask is not None:
            idx = 1 if gt_mask is not None else 0
            self._overlay_mask(axes[idx], image, pred_mask, self.cmap_pred, f"Prediction\n{metrics_str}")

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if display:
            plt.show()
        else:
            plt.draw()
        plt.close(fig)