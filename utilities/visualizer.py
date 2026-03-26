import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.Keypoints_detection.evaluation.evaluation_utils import get_gt_from_hrnet_label_files
import os
import numpy as np
import seaborn as sns


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


class TriangulationVisualizer:
    def __init__(self, output_dir='results/Keypoints_detection/inference_results/triangulation'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Skeleton for a 7-keypoint tool
        self.edges = [(0,1),(1,2),(2,3),(2,4)]
        self.tool_colors = ['red', 'blue'] 

    def reprojection_error_violin_plots(self, results_list, model_names=['HRNet', 'ViTPose']):
        """Plots and saves Violin plots and CSV metrics."""
        all_data = []
        is_comparison = len(results_list) == 2
        
        if is_comparison:
            title, save_name, x_axis = f"Comparison of the reprojection error: {model_names[0]} vs {model_names[1]}", "global_comparison", 'Model'
            for res, name in zip(results_list, model_names):
                for t_idx in range(len(res['reproj_err_l'])):
                    combined = np.concatenate([np.concatenate(res['reproj_err_l'][t_idx]), 
                                             np.concatenate(res['reproj_err_r'][t_idx])])
                    valid = combined[~np.isnan(combined)]
                    for err in valid: all_data.append({'Model': name, 'Error (pixels)': err})
        else:
            res = results_list
            title, save_name, x_axis = f"Comparison of the reprojection error per tool of model: ({model_names})", f"{model_names}_tools", 'Tool'
            for t_idx in range(len(res['reproj_err_l'])):
                combined = np.concatenate([np.concatenate(res['reproj_err_l'][t_idx]), 
                                         np.concatenate(res['reproj_err_r'][t_idx])])
                valid = combined[~np.isnan(combined)]
                for err in valid: all_data.append({'Tool': f'Tool {t_idx}', 'Error (pixels)': err})

        df = pd.DataFrame(all_data)
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=x_axis, y='Error (pixels)', data=df, inner="quartile", palette="muted")
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, f"{save_name}_violin.png"), dpi=300)
        
        summary = df.groupby(x_axis)['Error (pixels)'].agg(['mean', 'std', 'median', 'count']).round(3)
        summary.to_csv(os.path.join(self.output_dir, f"{save_name}_metrics.csv"))
        plt.show()
        return summary

    def plot_reprojections(self, img_l_path, img_r_path, results, triangulator, frame_name, show=False):
        """
        Uses the results from run_triangulation_pipeline for a single-frame path list.
        results: The dict returned by the pipeline.
        """
        img_l = cv2.imread(img_l_path)
        img_r = cv2.imread(img_r_path)
        
        # Extract data (frame 0 since we only passed one pair)
        pts_3d = results['tri_3d'] # List [Tool0_pts, Tool1_pts]
        preds_l = results['preds_l']
        preds_r = results['preds_r']

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        sides = [('left', img_l, preds_l), ('right', img_r, preds_r)]

        for i, (side_name, img, preds) in enumerate(sides):
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Loop through tools (max_tools)
            for t_idx in range(len(pts_3d)):
                # Get points for this specific tool at frame 0
                tool_3d = pts_3d[t_idx] 
                tool_2d_raw = preds[0, t_idx] 

                if not np.isnan(tool_3d).any():
                    # Use your triangulator's projection method
                    proj_2d = triangulator.project_points(np.array(tool_3d).squeeze(), side=side_name)
                    
                    #Plot Detected 
                    axes[i].scatter(tool_2d_raw[:, 0], tool_2d_raw[:, 1], 
                                   edgecolors='yellow', facecolors='none', s=40, label='Detected' if t_idx==0 else "")
                    
                    # Plot Reprojected 
                    axes[i].scatter(proj_2d[:, 0], proj_2d[:, 1], 
                                   c='red', marker='x', s=30, label='Reprojected' if t_idx==0 else "")
                    
                    # Draw Skeleton
                    for start, end in self.edges:
                        axes[i].plot([proj_2d[start, 0], proj_2d[end, 0]], 
                                     [proj_2d[start, 1], proj_2d[end, 1]], 
                                     color=self.tool_colors[t_idx], alpha=0.8, linewidth=2)

            axes[i].set_title(f"{side_name.upper()} View - Frame {frame_name}")
            axes[i].axis('off')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"repro_frame_{frame_name}.png"), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_3d_tools(self, pts_3d, frame_name, show=False):
        """Creates a 3D scatter plot of the triangulated tools."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for t_idx in range(pts_3d.shape[0]):
            tool_pts = pts_3d[t_idx] # (7, 3)
            if np.isnan(tool_pts).any(): continue
            
            # Plot joints
            ax.scatter(tool_pts[:, 0], tool_pts[:, 1], tool_pts[:, 2], c=self.tool_colors[t_idx], s=50)
            
            # Plot skeleton lines
            for start, end in self.edges:
                ax.plot([tool_pts[start, 0], tool_pts[end, 0]],
                        [tool_pts[start, 1], tool_pts[end, 1]],
                        [tool_pts[start, 2], tool_pts[end, 2]], color=self.tool_colors[t_idx], linewidth=2)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Reconstructed Tools - {frame_name}')
        plt.savefig(os.path.join(self.output_dir, f"3d_reconstruction_{frame_name}.png"))
        if show:
            plt.show()
        else:
            plt.close()

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