import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'src' directory 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from types import SimpleNamespace
import os
import torch
import yaml
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Work-directory-specific imports
from src.surgpose_dataset import SurgPoseDataset, SurgPoseDatasetOneInstance
from src.dataset_preprocessing import video_level_split
from src.utils import setup_logger, get_device, load_pretrained_HRNet

# Hrnet-package-specific imports
from hrnet_config import cfg , update_config
from core.loss import JointsMSELoss


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss=0
    for images, targets, weights, _ in tqdm(dataloader, desc="Training", unit="batch" ):
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
        for images, targets, weights, _  in tqdm(dataloader, desc="Validation", unit="batch" ):
            images= images.to(device)
            targets= targets.to(device)
            weights = weights.to(device)
            
            preds= model(images)
            loss = criterion(preds,targets,weights)         
            total_loss+= loss.item()
    return  total_loss/len(dataloader)
    
def main():
    parser = argparse.ArgumentParser(description='Train HRNet for Surgical Keypoints')
    parser.add_argument('--cfg_path', help='experiment config path', required=True, type=str)
    # # Add toggle for the "One Instance" (Cropped/YOLO) dataset
    # parser.add_argument('--one_instance', action='store_true', help='Use the cropped one-instance dataset')
    arguments = parser.parse_args()

    # read amd import the config file
    cfg_file = arguments.cfg_path
    args = SimpleNamespace(
        cfg=cfg_file,
        opts=[],
        modelDir='',
        logDir='',
        dataDir='',
        prevModelDir=''
    )
    update_config(cfg, args)
    
    model_weights= cfg.MODEL.EXTRA.ORIGINAL_PAPER_WEIGHTS
    main_dir= cfg.MODEL.EXTRA.SAVE.EXPERIMENT_DIR
    one_instance = cfg.MODEL.EXTRA.ONE_INSTANCE
    
    # Get time stamp, initiate checkpoint's file path and create log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir= os.path.join(main_dir,f'training_checkpoints{timestamp}')
    log_dir = os.path.join(main_dir, 'logs')
    logger = setup_logger(log_dir,log_name=f'finetuning_{timestamp}')
    logger.info("Starting HRNet fine-tuning...")
    
    # Setup loss file paths and create the file
    loss_file = os.path.join(log_dir, f"losses_per_epoch_{timestamp}.csv")
        # Create CSV and write header if it doesn't exist
    if not os.path.exists(loss_file):
        with open(loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    # Get training hyperparameters and info
    batch_size = cfg.MODEL.EXTRA.TRAINING.BATCH_SIZE
    num_epochs= cfg.MODEL.EXTRA.TRAINING.NUM_EPOCHS
    state = cfg.MODEL.EXTRA.TRAINING.ALREADY_FINETUNED
    lr = float(cfg.MODEL.EXTRA.TRAINING.LR)
    save_int= cfg.MODEL.EXTRA.TRAINING.SAVE_INT
    
    # Load the pretrained HRNet model 
    model = load_pretrained_HRNet(cfg, model_weights, finetuned=state)
    joints = cfg.MODEL.NUM_JOINTS
    device = get_device()
    model = model.to(device)
    
    # Log teh model info
    model_name = os.path.splitext(os.path.basename(model_weights))[0]
    logger.info(f"Finetuning model: {model_name} with {joints} heatmaps")
    logger.info(f'Model initial state: already finetuned = {state}')
    logger.info(f"Using device: {device}")
   

    # Get input dataset information and log it
    in_height, in_width= cfg.MODEL.IMAGE_SIZE[0],cfg.MODEL.IMAGE_SIZE[1]
    H_h, W_h= cfg.MODEL.HEATMAP_SIZE[0],cfg.MODEL.HEATMAP_SIZE[1]
    sigma=cfg.MODEL.SIGMA
    logger.info(f'input_size: H={in_height}, W={in_width}')

    # get the image data path and the suitable annotation data path
    data_root= cfg.MODEL.EXTRA.DATA.ROOT
    frames_dir= os.path.join(data_root,'extracted_frames')
    if one_instance:
        annotations_dir= os.path.join(data_root,'extracted_bboxes_kpts')
    else:
        annotations_dir= os.path.join(data_root,'extracted_keypoints')
    
    # Load the video level split file and get video IDs for each category
    output_split_file= os.path.join(data_root,'video_split.yaml')
    with open(output_split_file, "r") as f:
        splits = yaml.safe_load(f)
    train_video_list = splits['train']
   
    val_video_list =splits['val']

    # Define data augmentation transforms if augmentation is included in this training
    if cfg.MODEL.EXTRA.TRAINING.AUGMENTATION:
        train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.OneOf([
            A.RandomResizedCrop(in_height, in_width, scale=(0.5, 1.0), p=0.5), 
            A.Resize(in_height,in_width,p=0.5),  
        ], p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()]
        , keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        train_transforms= None
    
    # Log the transforms 
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
    
    # Create the suitable dataset instance
    if  one_instance:
        train_dataset = SurgPoseDatasetOneInstance(frames_dir,annotations_dir, input_size=(in_width,in_height), heatmap_size=(W_h,H_h),transform=train_transforms,sigma=sigma, num_joints=joints,video_list=train_video_list)
        val_dataset = SurgPoseDatasetOneInstance(frames_dir,annotations_dir,input_size=(in_width,in_height), heatmap_size=(W_h,H_h), sigma=sigma,num_joints=joints, video_list=val_video_list)
    else:
        train_dataset = SurgPoseDataset(frames_dir,annotations_dir,video_list=train_video_list, input_size=(in_width,in_height), heatmap_size=(W_h,H_h),transform=train_transforms,sigma=sigma)
        val_dataset = SurgPoseDataset(frames_dir,annotations_dir,video_list=val_video_list,input_size=(in_width,in_height), heatmap_size=(W_h,H_h), sigma=sigma)
    # Create dataloaders
    train_loader= DataLoader(train_dataset, batch_size= batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss and optimizer and log the info
    criterion = JointsMSELoss(use_target_weight=False) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    logger.info(f'batch size: {batch_size}, Lr: {lr}, number of epochs:{num_epochs}')
    
    loss_name = criterion.__class__.__name__
    params = {k: v for k, v in criterion.__dict__.items() if not k.startswith('_')}
    logger.info(f"Loss: {loss_name}, parameters: {params}")

    # Training and validation loop
    
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