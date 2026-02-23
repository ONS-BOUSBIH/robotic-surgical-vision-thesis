import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'src' directory 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Train ViTPose')
    parser.add_argument('--cfg', help='Path to ViTPose config file', required=True)
    args = parser.parse_args()

    # Load the MMPose config
    cfg = Config.fromfile(args.cfg)
    # Build the Runner
    runner = Runner.from_cfg(cfg)
    # Start Training
    runner.train()

if __name__ == '__main__':
    main()