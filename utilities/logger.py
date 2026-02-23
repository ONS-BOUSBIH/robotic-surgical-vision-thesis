import csv
from datetime import datetime
import os

def log_evaluation_results_kpts(model_name, weights_path, metrics, log_path="evaluation_logs.csv", printing= True, saving= True):
    """
    Prints results and appends them to a csv log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Header 
    header = ["Timestamp", "Model", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95", "Weights_Path"]

    row = [
        timestamp,
        model_name,
        metrics.get("num_images"),
        metrics.get("num_valid"),
        f"{metrics.get('precision', 0):.4f}",
        f"{metrics.get('recall', 0):.4f}",
        f"{metrics.get('map50', 0):.4f}",
        f"{metrics.get('map50_95', 0):.4f}",
        weights_path
    ]

    # Prints
    if printing:
        print("\n" + "="*50)
        print(f"{model_name.upper()} TEST EVALUATION RESULTS")
        print("-" * 50)
        
        for i in range(3, len(header)):
            print(f"{header[i]:<15}: {row[i]}")
        print("="*50)

    # Save
    if saving:
        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        
        print(f"Log updated: {os.path.abspath(log_path)}")