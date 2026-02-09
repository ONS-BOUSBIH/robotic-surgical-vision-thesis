#load trained model and run inference on a test set
from training import *
import torch
from torch.utils.data import DataLoader
from surgpose_dataset import SurgPoseDataset
from core.inference import get_max_preds
from finetuning_utils import WeightedJointsMSELoss, custom_get_max_preds

def inference(model,device ,data_root, split_file,input_size,heatmap_size ,custom_arg_max= True, threshold=0.2 ):
    #cfg_file = '/srv/homes/onbo10/thesis_Ons/HRNet-experiments/HRNet_finetuned/w32_256x192_adam_lr1e-3_out14-finetune.yaml'
    #model_weights="./Experiment1/training_chekpoints2025-11-07_12-30-59/model_epoch30.pth"
    model.to(device)
    model.eval()
    
   # data_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted' 
    frames_dir= os.path.join(data_root,'extracted_frames')
    keypoints_dir= os.path.join(data_root,'extracted_keypoints')
    #split_file= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted/video_split.yaml'
    with open(split_file, "r") as f:
        splits = yaml.safe_load(f)
    test_video_list = splits['test']    
    test_dataset = SurgPoseDataset(frames_dir,keypoints_dir,video_list=test_video_list, input_size=input_size, heatmap_size=heatmap_size)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    all_preds = []
    all_targets=[]
    all_gt_masks=[]
    all_maxvals=[]

    with torch.no_grad():
      
        for images, targets, masks in tqdm(test_loader):
            images = images.to(device)
            #print('input', images.shape)
            targets = targets.to(device)
           # gt_masks= masks.to(device)
            outputs = model(images)
            #print('output:', outputs.shape)
            outputs = outputs.cpu().numpy()
            targets= targets.cpu().numpy()
            if custom_arg_max:
                preds, maxval= custom_get_max_preds(outputs, threshold) 
            else:
                preds, maxval= get_max_preds(outputs) 
            
            gts, _ =  get_max_preds(targets) 
            
            all_preds.extend(preds)
            all_targets.extend(gts)
            all_gt_masks.extend(masks.cpu().numpy())
            all_maxvals.extend(maxval)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_gt_masks = np.array(all_gt_masks).squeeze()
        all_maxvals  = np.array(all_maxvals).squeeze()

    return  all_preds, all_targets, all_gt_masks, all_maxvals


if __name__== "__main__":
    cfg_file = '/srv/homes/onbo10/thesis_Ons/HRNet-experiments/HRNet_finetuned/w32_256x192_adam_lr1e-3_out14-finetune.yaml'
    model_weights="./Experiment1/training_chekpoints2025-11-07_12-30-59/model_epoch30.pth"
    data_root= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_right_test' 
    split_file= '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted/video_split.yaml'
    model= load_pretrained_HRNet(cfg_file,model_weights, finetuned=True)
    device= get_device()
    all_preds, all_targets, all_gt_masks, all_maxvals= inference(model,device,data_root, split_file)
