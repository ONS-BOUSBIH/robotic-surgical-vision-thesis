import os
import torch
import logging

# from hrnet_config import cfg , update_config
from models import pose_hrnet

def setup_logger(log_dir, log_name="training.log"):
    """Set up a logger that writes to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger("HRNet_Training")
    logger.setLevel(logging.INFO)
    logger.handlers = [] 

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(file_formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_pretrained_HRNet(cfg, model_weights, finetuned= False):
    """loads a pretrained weights to a modified HRNet structure and removes last layer weights to fit the new architecture"""
    assert os.path.exists(model_weights), "Please download the COCO pretrained weights."
    # args = SimpleNamespace(
    #     cfg=cfg_file,
    #     opts=[],
    #     modelDir='',
    #     logDir='',
    #     dataDir='',
    #     prevModelDir=''
    # )
    # update_config(cfg, args)
    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    checkpoint = torch.load(model_weights)
    ### CHANGES THAT NEED TO BE APPLIED TO THE PRETRAINED MODEL
    # removing final layer weights from checkpoints if the checkpoints are not of an already finedtuned model
    if not finetuned:
        print('true')
        checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('final_layer.')}
    #model = pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint,  strict=False)

    return model