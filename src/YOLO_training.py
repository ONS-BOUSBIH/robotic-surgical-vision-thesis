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

