import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self, log_path, save_dir="plots"):
        self.df = pd.read_csv(log_path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_losses(self, train_col='train/loss', val_col='val/loss', title="Training History"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[train_col], label='Train Loss', color='#1f77b4', lw=2)
        plt.plot(self.df[val_col], label='Val Loss', color='#ff7f0e', lw=2)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        save_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Plot saved to {save_path}")


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
        ax.set_title(f"Instance: {name}")
        ax.legend()
        plt.axis('off')
        
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"), bbox_inches='tight')
        plt.show()