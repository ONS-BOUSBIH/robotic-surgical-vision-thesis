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


def evaluate_MAE_and_compare(results_list, model_names=['HRNet', 'ViTPose'], output_dir='results/Keypoints_detection/inference_results/triangulation'):
    """
    If results_list has one dict: Plots Tool 0 vs Tool 1 for that model.
    If results_list has two dicts: Plots HRNet vs ViTPose (merging tools).
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    # Case 1: Comparing Two Models
    if len(results_list) == 2:
        title = "Model Comparison: HRNet vs ViTPose"
        save_name = "global_model_comparison"
        x_axis_col = 'Model'
        
        for res_dict, m_name in zip(results_list, model_names):
            # Flatten all tools into one for this model
            all_errs = []
            for t_idx in range(len(res_dict['reproj_err_l'])):
                l = np.concatenate(res_dict['reproj_err_l'][t_idx])
                r = np.concatenate(res_dict['reproj_err_r'][t_idx])
                all_errs.append(np.concatenate([l, r]))
            
            combined = np.concatenate(all_errs)
            valid_errs = combined[~np.isnan(combined)]
            for err in valid_errs:
                all_data.append({'Model': m_name, 'Error (pixels)': err})

    # Case 2: Standard Tool Comparison (Single Model)
    else:
        res_dict = results_list[0]
        title = f"Tool Comparison for {model_names[0]}"
        save_name = f"{model_names[0]}_tool_comparison"
        x_axis_col = 'Tool'
        
        for t_idx in range(len(res_dict['reproj_err_l'])):
            l = np.concatenate(res_dict['reproj_err_l'][t_idx])
            r = np.concatenate(res_dict['reproj_err_r'][t_idx])
            combined = np.concatenate([l, r])
            valid_errs = combined[~np.isnan(combined)]
            for err in valid_errs:
                all_data.append({'Tool': f'Tool {t_idx}', 'Error (pixels)': err})

    df = pd.DataFrame(all_data)

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.violinplot(x=x_axis_col, y='Error (pixels)', data=df, inner="quartile", palette="Set2")
    plt.title(title, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save Plot
    plot_path = os.path.join(output_dir, f"{save_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Metrics Summary
    summary = df.groupby(x_axis_col)['Error (pixels)'].agg(['mean', 'std', 'median', 'count']).round(3)
    
    # Save Metrics to CSV
    csv_path = os.path.join(output_dir, f"{save_name}_metrics.csv")
    summary.to_csv(csv_path)
    
    print(f"Results saved to {output_dir}")
    return summary