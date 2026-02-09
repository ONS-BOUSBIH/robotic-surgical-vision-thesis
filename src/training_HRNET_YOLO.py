import sys
project_root = '/srv/homes/onbo10/thesis_Ons'
if project_root not in sys.path:
    sys.path.append(project_root)
from models import pose_hrnet
from hrnet_config import cfg , update_config
from HRNet_YOLO.HRNet_one_instance.surgpose_dataset_one_instance import SurgPoseDatasetOneInstance
from core.loss import JointsMSELoss
import os
import torch
import numpy as np
from types import SimpleNamespace
import yaml
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import csv



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

def load_pretrained_HRNet(cfg_file, model_weights, finetuned= False):
    """loads a pretrained weights to a modified HRNet structure and removes last layer weights to fit the new architecture"""
    assert os.path.exists(model_weights), "Please download the COCO pretrained weights."
    args = SimpleNamespace(
        cfg=cfg_file,
        opts=[],
        modelDir='',
        logDir='',
        dataDir='',
        prevModelDir=''
    )
    update_config(cfg, args)
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

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss=0
    for images, targets, weights, side in tqdm(dataloader, desc="Training", unit="batch" ):
        images= images.to(device)
        targets= targets.to(device)
        weights = weights.to(device)
        
        preds= model(images)
        loss = criterion(preds,targets,weights)
        #print('output_shape', preds.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
    
    return total_loss/len(dataloader)

def validate(model,dataloader,criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, targets, weights , side in tqdm(dataloader, desc="Validation", unit="batch" ):
            images= images.to(device)
            targets= targets.to(device)
            weights = weights.to(device)
            
            preds= model(images)
            loss = criterion(preds,targets,weights)         
            total_loss+= loss.item()
    return  total_loss/len(dataloader)
    
def main():


    cfg_file = '/srv/homes/onbo10/thesis_Ons/HRNet_experiments/HRNet_finetuned/w32_256x192_adam_lr1e-3_out14-finetune.yaml' #'/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/HRNet_one_instance/w32_256x192_adam_lr1e-3_out7-finetune.yaml'
    model_weights='/srv/homes/onbo10/thesis_Ons/HRNet-Human-Pose-Estimation/pose_hrnet_w32_256x192.pth'  
    main_dir='/srv/homes/onbo10/thesis_Ons/HRNet_YOLO/HRNet_one_instance/Experiment2'
    split_file = '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/video_split.yaml'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir= os.path.join(main_dir,f'training_chekpoints{timestamp}')
    log_dir = os.path.join(main_dir, 'logs')
    logger = setup_logger(log_dir,log_name=f'finetuning_{timestamp}')
    logger.info("Starting HRNet fine-tuning...")
    
    loss_file = os.path.join(log_dir, f"losses_per_epoch_{timestamp}.csv")
    train_transforms= None
    # Create CSV and write header if it doesn't exist
    if not os.path.exists(loss_file):
        with open(loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])


    batch_size = 32
    num_epochs= 200
    state = False
    LR= 1e-3
    save_int= 100
   
    model = load_pretrained_HRNet(cfg_file, model_weights, finetuned=state)
    joints = cfg.MODEL.NUM_JOINTS
    print(joints)
    device = get_device()
    model = model.to(device)
 
    model_name = os.path.splitext(os.path.basename(model_weights))[0]
    logger.info(f"Finetuning model: {model_name} with {joints} heatmaps")
    logger.info(f'Model initial state: already finetuned = {state}')
    logger.info(f"Using device: {device}")
   

    in_height, in_width= cfg.MODEL.IMAGE_SIZE[0],cfg.MODEL.IMAGE_SIZE[1]
    H_h, W_h= cfg.MODEL.HEATMAP_SIZE[0],cfg.MODEL.HEATMAP_SIZE[1]
   
    sigma=cfg.MODEL.SIGMA
    logger.info(f'input_size: H= {in_height}, W= {in_width}')

    data_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right'
    frames_dir= os.path.join(data_root,'extracted_frames')
    annotations_dir= os.path.join(data_root,'extracted_bboxes_kpts')
    
    
    #### Generate a video level split and load datasets
    #video_level_split(frames_dir, output_split_file,train=0.7,val=0.15,seed=42)
    
    with open(split_file, "r") as f:
        splits = yaml.safe_load(f)
    train_video_list = splits['train']
   
    val_video_list =splits['val']

    #Define data augmentation transforms
    train_transforms= None
    
    # Logging the transforms 
    if train_transforms:
        logger.info(' Data augmentation transforms: ')
        for t in train_transforms.transforms:
            # Some transforms like OneOf contain a nested list of transforms
            if isinstance(t, A.OneOf):
                logger.info(f"OneOf with probability {t.p}:")
                for sub_t in t.transforms:
                    logger.info(f"   - {sub_t.__class__.__name__}: {sub_t.get_params()}, p={sub_t.p}")
            else:
                logger.info(f"{t.__class__.__name__}: {t.get_params()}, p={t.p}")
    else:
        logger.info('No data augmentation')


    train_dataset = SurgPoseDatasetOneInstance(frames_dir,annotations_dir, input_size=(in_width,in_height), heatmap_size=(W_h,H_h),transform=train_transforms,sigma=sigma, num_joints=joints,video_list=train_video_list)
    val_dataset = SurgPoseDatasetOneInstance(frames_dir,annotations_dir,input_size=(in_width,in_height), heatmap_size=(W_h,H_h), sigma=sigma,num_joints=joints, video_list=val_video_list)
    
    train_loader= DataLoader(train_dataset, batch_size= batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = JointsMSELoss(use_target_weight=False) 
    #criterion= WeightedJointsMSELoss(visible_weight=1.0, invisible_weight=0.2)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    logger.info(f'batch size: {batch_size}, Lr: {LR}, number of epochs:{num_epochs}')
    
    loss_name = criterion.__class__.__name__
    params = {k: v for k, v in criterion.__dict__.items() if not k.startswith('_')}
    logger.info(f"Loss: {loss_name}, parameters: {params}")


    best_val_loss = float('inf')

    for epoch in tqdm(range(1,num_epochs+1), desc='Training epochs', unit='epoch'):
        train_loss= train(model,train_loader,optimizer,criterion,device)
        val_loss = validate(model,val_loader,criterion, device)
        logger.info(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss} | Val Loss: {val_loss}")

        # save losses in csv file
        with open(loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])
            
        # save checkpoints
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        
        #Save ckpts for best validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_file = os.path.join(ckpt_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': train_loss,
                'avg_val_loss': val_loss,
            }, best_file)
            logger.info(f"Best checkpoint saved: {best_file}")

        # save ckpts periodically
        if epoch % save_int == 0:
            periodic_file = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': train_loss,
                'val_loss': val_loss,
            }, periodic_file)
            logger.info(f'Periodic checkpoint saved: {periodic_file}')




if __name__== '__main__':
    main()