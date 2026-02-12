import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'src' directory 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to YOLO config yaml')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Load the base model
    model = YOLO(config_data['model']) 

    # Train using the settings defined in the config file
    model.train(cfg=args.cfg)

if __name__ == '__main__':
    main()


# model = YOLO("yolov8n.pt")
# data_doc='/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/yolo_formated_surgpose/data.yaml'
# results = model.train(
#     data=data_doc,
#     epochs=50,
#     imgsz=960,
#     batch=16,
#     device=2,     
#     workers=4,
#     project="runs_yolo",
#     name="surgpose_exp1"
# )

# model = YOLO("yolov8x-pose.pt")

# data_yaml= '/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/yolo_formated_surgpose_kpts/surgpose_pose.yaml'
# model.train(
#     data=data_yaml,
#     epochs=50,
#     imgsz=640,         
#     batch=16,
#     device=1,         
#     workers=4,
#     cache=True,
#     name="yolov8pose_surgpose"
# )
