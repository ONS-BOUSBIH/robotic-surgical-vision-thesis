
import torch
from torch.utils.data import DataLoader
from evaluation_dataset import SurgPoseDatasetOneInstanceInference
import numpy as np
from ultralytics.utils.metrics import kpt_iou
from ultralytics.utils.ops import xyxy2xywh
from core.inference import get_max_preds
import torch
import numpy as np
from ultralytics.utils.metrics import ap_per_class
from core.inference import get_max_preds
import re
import yaml
import os
from tqdm import tqdm
import cv2

def process_batch(pred_kpts, gt_kpts, gt_areas, sigmas, confidences, iou_thresholds):
    """
    Matches predictions to ground truths based on OKS.
    pred_kpts: [M, n_joints, 3] (x, y, conf)
    gt_kpts: [N, n_joints, 3] (x, y, visibility)
    gt_areas: [N] (areas of GT objects)
    """
    num_thresholds = len(iou_thresholds)
    num_preds = pred_kpts.shape[0]
    # correct_matrix: [num_preds, 10] (True/False for each OKS threshold)
    correct_matrix = torch.zeros(num_preds, num_thresholds, dtype=torch.bool)
    
    if num_preds == 0:
        return correct_matrix

    # 1. Compute OKS Matrix [N, M]
    oks = kpt_iou(gt_kpts, pred_kpts, gt_areas, sigmas)

    # 2. Matching Logic
    for i, threshold in enumerate(iou_thresholds):
        # Find matches where OKS > threshold
        matches = torch.where(oks > threshold) 
        if matches[0].shape[0] > 0:
            match_data = torch.stack([matches[0], matches[1], oks[matches]], 1).cpu().numpy()
            if match_data.shape[0] > 1:
                # Sort matches by highest OKS
                match_data = match_data[match_data[:, 2].argsort()[::-1]]
                # Ensure one GT matches only one Pred (and vice versa)
                match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
                match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
            
            correct_matrix[match_data[:, 1].astype(int), i] = True
            
    return correct_matrix

def evaluate_cropped_HRNET(model, test_loader,device ,SIGMAS, IOU_THRESHOLDS, w_m=64.0, h_m=48.0):
    
    stats = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, targets, targets_w,gt_kpts, gt_bboxes,crops, gt_cls = batch
            images =images.to(device)
            gt_kpts = gt_kpts.to(device)    # [B, 7, 3]
            gt_bboxes = gt_bboxes.to(device)  # [B, 4]
            crops = crops.to(device)    # [B, 4]
            
            outputs = model(images) 
            
            #Get coordinates of keypoints and confidence
            preds_local, maxvals = get_max_preds(outputs.cpu().numpy())
            
            #Resize and rescale
            preds_global = np.zeros_like(preds_local)
            for i in range(images.shape[0]):
                c_x1, c_y1, c_x2, c_y2 = crops[i].cpu().numpy()
                crop_w, crop_h = c_x2 - c_x1, c_y2 - c_y1
                
                #from heatmap space to original image pixels
                preds_global[i, :, 0] = (preds_local[i, :, 0] * (crop_w / w_m)) + c_x1
                preds_global[i, :, 1] = (preds_local[i, :, 1] * (crop_h / h_m)) + c_y1

            pred_kpts = torch.from_numpy(np.concatenate([preds_global, maxvals], axis=2)).to(device)

            #Calculate Stats for each image in batch
            for i in range(len(images)):
                pk = pred_kpts[i]      # [7, 3]
                gk = gt_kpts[i]        # [7, 3]
                gb = gt_bboxes[i]      # [4]
                
                # Calculate OKS Area (using original bbox)
                area = (gb[2] - gb[0]) * (gb[3] - gb[1]) 
                area = area.unsqueeze(0).to(device) 

                # Confidence is the mean of the keypoint scores
                obj_conf = pk[:, 2].mean().unsqueeze(0) 
                
                # process_batch returns a (1, 10) boolean matrix
                tp_matrix = process_batch(pk.unsqueeze(0), gk.unsqueeze(0), area, SIGMAS, obj_conf, IOU_THRESHOLDS)
                
                # Store results: (tp, conf, pred_cls, gt_cls)
                stats.append((tp_matrix.cpu(), obj_conf.cpu(), torch.zeros(1), torch.zeros(1)))

    # mAP Calaculation
    if len(stats) > 0:
        # Concatenate all image results
        tp, conf, pcls, gcls = [torch.cat(x, 0).numpy() for x in zip(*stats)]
       
        results = ap_per_class(tp, conf, pcls, gcls, plot=False)

        tp_res, fp_res, p, r, f1, ap, unique_classes = results[0], results[1], results[2], results[3], results[4], results[5], results[6]

        map50_95 = ap.mean()
        map50 = ap[:, 0].mean() 
        return len(stats), p.mean(), r.mean(), map50, map50_95
    else:
        print('No detections')
        return(None)

def evaluate_YOLO(yolo_model, dataloader, device,SIGMAS_YOLO, IOU_THRESHOLDS, valid_keys, ann_root):
    yolo_stats = []
    total_valid_instances = 0

    for batch in dataloader:
       
        img_path = batch["img_path"][0] 
        img_name = os.path.basename(img_path)
        
        m = re.search(r'vid_(\d+)', img_name)
        if m is None:
            continue
        video_id = str(m.group(1))
    
        # Run YOLO Inference
        img_list = [img.numpy().astype(np.uint8) for img in batch["img"].permute(0, 2, 3, 1)]
        results = yolo_model(img_list, verbose=False)
        h0, w0 = batch["h_w_orig"][0].item(), batch["h_w_orig"][1].item()

        # Load yaml annotations to get the exact Object IDs for this image
       
        yaml_file= os.path.basename(img_path.replace(".jpg", ".yaml"))
        
        # construct the corresponding yaml annotation file from the Surgpose Dataset adapted to HNRET trainig
        yaml_path = os.path.join(ann_root,video_id,yaml_file)
    
        gt_bboxes_list = []
        gt_kpts_list = []
        ignored_bboxes_list = []

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                ann = yaml.safe_load(f)
            
            for obj in ann["objects"]:
                key = f"{video_id}/{img_name}_{obj['id']}"
                
                # Match against the valid instances for evaluation (from HRNet evaluation)
                if key in valid_keys:
                    gt_bboxes_list.append(obj["bbox"])
                    # Add visibility column to keypoints [7, 3]
                    kpts = np.array(obj["keypoints"])
                    vis = np.array(obj["visibility"]).reshape(-1, 1)
                    gt_kpts_list.append(np.concatenate([kpts, vis], axis=1))
                else:
                    # This instance was rejected by HRNet - we must ignore YOLO preds here
                    ignored_bboxes_list.append(obj["bbox"])

        # If this frame has no instances that HRNet was evaluated on, skip it to keep image counts aligned
        if not gt_bboxes_list:
            continue

        gt_bboxes = torch.tensor(gt_bboxes_list, device=device, dtype=torch.float32)
        gt_kpts = torch.tensor(gt_kpts_list, device=device, dtype=torch.float32)
        ignored_bboxes = torch.tensor(ignored_bboxes_list, device=device, dtype=torch.float32)
        
        total_valid_instances += len(gt_bboxes)

        # Process YOLO Predictions
        for r in results:
            pk = r.keypoints.data.clone()
            pb = r.boxes.xyxy.clone() 
            pc = r.boxes.conf.clone()

            if pk.shape[0] > 0:
                # Scale to original pixels
                pk[:, :, 0] = r.keypoints.xyn[:, :, 0] * w0
                pk[:, :, 1] = r.keypoints.xyn[:, :, 1] * h0
                
                #Filter Predictions
                # Remove any prediction that overlaps with an instance HRNet ignored
                keep_pred = torch.ones(len(pb), dtype=torch.bool).to(device)
                
                if len(ignored_bboxes) > 0:
                    from ultralytics.utils.metrics import box_iou
                    # Calculate IoU between YOLO predictions and IGNORED GT boxes
                    iou_ignored = box_iou(pb, ignored_bboxes) 
                    # If prediction is on an ignored tool, drop it from stats
                    is_on_ignored = (iou_ignored > 0.4).any(dim=1) 
                    keep_pred = ~is_on_ignored

                pk_filtered = pk[keep_pred]
                pc_filtered = pc[keep_pred]
            else:
                pk_filtered = torch.zeros((0, 7, 3), device=device)
                pc_filtered = torch.zeros(0, device=device)

            # Compute OKS and find TP instances for different thresholds
            # Compute GT bboxes area as scale for OKS computation
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            
            # TP matches matrix
            tp_matrix = process_batch(pk_filtered, gt_kpts, gt_areas, SIGMAS_YOLO, pc_filtered, IOU_THRESHOLDS)
            
            yolo_stats.append((
                tp_matrix.cpu(), 
                pc_filtered.cpu(), 
                torch.zeros(len(pc_filtered)), 
                torch.zeros(len(gt_kpts))
            ))

   
    tp, conf, pcls, gcls = [torch.cat(x, 0).numpy() for x in zip(*yolo_stats)]
    results_custom = ap_per_class(tp, conf, pcls, gcls)

    tp_res, fp_res, p, r, f1, ap, unique_classes = results_custom[0], results_custom[1], results_custom[2], results_custom[3], results_custom[4], results_custom[5], results_custom[6]
    map50_95 = ap.mean()
    map50 = ap[:, 0].mean() 
    return len(yolo_stats), p.mean(),r.mean(), map50, map50_95 , total_valid_instances

def evaluate_HRNet_full_image(model, dataloader, device,SIGMAS, IOU_THRESHOLDS, valid_keys, ann_root, w_m=64.0, h_m=48.0):

    yolo_stats_hrnet_full = []
    total_valid_instances = 0

    for batch in dataloader:
        img_path = batch["meta"]["img_path"][0] 
        img_name = os.path.basename(img_path)
        m = re.search(r'vid_(\d+)', img_name)
        if m is None: continue
        video_id = str(m.group(1))
        
        # HRNet Full-Frame Inference
        img_tensor = batch["img"].to(device) 
        with torch.no_grad():
            output_heatmaps = model(img_tensor).cpu().numpy() 

        preds, maxvals = get_max_preds(output_heatmaps)

        # Scale and Group into 2 Instances
        h0, w0 = batch["meta"]["h_w_orig"][0]
        h_hm, w_hm = output_heatmaps.shape[2], output_heatmaps.shape[3]
        
        # Scale heatmap coords to original resolution
        preds[:, :, 0] *= (w0.item() / w_hm)
        preds[:, :, 1] *= (h0.item()/ h_hm)

        # Combine x,y with confidence (maxvals) into [2, 7, 3]
        inst1 = np.concatenate([preds[0, 0:7, :], maxvals[0, 0:7, :]], axis=1)
        inst2 = np.concatenate([preds[0, 7:14, :], maxvals[0, 7:14, :]], axis=1)
        
        all_preds = torch.tensor(np.stack([inst1, inst2]), device=device, dtype=torch.float32)
        pred_confs = all_preds[:, :, 2].mean(dim=1) 

        #  Load GT and filter corrupt instances
        yaml_file = os.path.basename(img_path.replace(".jpg", ".yaml"))
        yaml_path = os.path.join(ann_root, video_id, yaml_file)
        
        gt_bboxes_valid, gt_kpts_valid, corrupt_bboxes = [], [], []

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                ann = yaml.safe_load(f)
            for obj in ann["objects"]:
                key = f"{video_id}/{img_name}_{obj['id']}"
                if key in valid_keys: 
                    gt_bboxes_valid.append(obj["bbox"])
                    kpts = np.array(obj["keypoints"])
                    vis = np.array(obj["visibility"]).reshape(-1, 1)
                    gt_kpts_valid.append(np.concatenate([kpts, vis], axis=1))
                else:
                    corrupt_bboxes.append(obj["bbox"])

        if not gt_bboxes_valid:
            continue

        gt_kpts = torch.tensor(gt_kpts_valid, device=device, dtype=torch.float32)
        gt_bboxes = torch.tensor(gt_bboxes_valid, device=device, dtype=torch.float32)
        total_valid_instances += len(gt_kpts)

        # Ignore predictions on corrupt Bboxes
        keep_pred = torch.ones(len(all_preds), dtype=torch.bool).to(device)
        for i, pred in enumerate(all_preds):
            pred_center = pred[:, :2].mean(dim=0) 
            for c_bbox in corrupt_bboxes:
                if c_bbox[0] <= pred_center[0] <= c_bbox[2] and \
                c_bbox[1] <= pred_center[1] <= c_bbox[3]:
                    keep_pred[i] = False
                    break

        all_preds_filtered = all_preds[keep_pred]
        pred_confs_filtered = pred_confs[keep_pred]

        # Matching
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        
        tp_matrix = process_batch(all_preds_filtered, gt_kpts, gt_areas, SIGMAS, pred_confs_filtered, IOU_THRESHOLDS)
        
        yolo_stats_hrnet_full.append((
            tp_matrix.cpu(), 
            pred_confs_filtered.cpu(), 
            torch.zeros(len(pred_confs_filtered)), 
            torch.zeros(len(gt_kpts))
        ))
    
    tp, conf, pcls, gcls = [torch.cat(x, 0).numpy() for x in zip(*yolo_stats_hrnet_full)]
    results_custom = ap_per_class(tp, conf, pcls, gcls)
    tp_res, fp_res, p, r, f1, ap, unique_classes = results_custom[0],results_custom[1], results_custom[2], results_custom[3], results_custom[4], results_custom[5], results_custom[6]
    map50_95 = ap.mean()
    map50 = ap[:, 0].mean() 

    return len(yolo_stats_hrnet_full), p.mean(),r.mean(), map50, map50_95 , total_valid_instances

def evaluate_ViTPose_custom(model, test_loader, device, SIGMAS, IOU_THRESHOLDS):
    stats = []
    model.eval()
    
    # Note: ViTPose (MMPose 1.x) uses a data_preprocessor to handle normalization
    # Ensure your test_loader is yielding data compatible with MMPose PackPoseInputs
    
    with torch.no_grad():
        for batch in test_loader:
            # 1. Move data to device using the model's preprocessor
            # This handles normalization (mean/std) automatically
            data = model.data_preprocessor(batch, False)
            
            # 2. Run Inference
            # Returns a list of PoseDataSample objects
            results = model.predict(**data)
            
            for i, result in enumerate(results):
                # Extract predicted keypoints [K, 2] and scores [K]
                # These are already decoded from heatmaps to pixel coordinates
                pred_instances = result.pred_instances
                pk_coords = pred_instances.keypoints[0] # [K, 2]
                pk_scores = pred_instances.keypoint_scores[0] # [K]
                
                # Combine into [K, 3] (x, y, conf) to match your 'pk' format
                pk = torch.cat([
                    torch.from_numpy(pk_coords), 
                    torch.from_numpy(pk_scores).unsqueeze(-1)
                ], dim=1).to(device)
                
                # 3. Get Ground Truth from the result object
                # MMPose stores GT in the same data sample
                gt_instances = result.gt_instances
                gk = torch.from_numpy(gt_instances.keypoints[0]).to(device) # [K, 2]
                gv = torch.from_numpy(gt_instances.keypoints_visible[0]).to(device) # [K]
                
                # Format GT as [K, 3] (x, y, visibility)
                gk_formatted = torch.cat([gk, gv.unsqueeze(-1)], dim=1)
                
                # 4. Area Calculation (using GT Bbox)
                gt_bboxes = gt_instances.bboxes[0] # [x1, y1, x2, y2]
                area = (gt_bboxes[2] - gt_bboxes[0]) * (gt_bboxes[3] - gt_bboxes[1])
                area = torch.tensor([area], device=device)

                # Object Confidence (mean of kpt scores, as per your HRNet code)
                obj_conf = pk[:, 2].mean().unsqueeze(0)
                
                # 5. Your custom Matching Logic
                tp_matrix = process_batch(
                    pk.unsqueeze(0), 
                    gk_formatted.unsqueeze(0), 
                    area, 
                    SIGMAS, 
                    obj_conf, 
                    IOU_THRESHOLDS
                )
                
                stats.append((tp_matrix.cpu(), obj_conf.cpu(), torch.zeros(1), torch.zeros(1)))

    # mAP Calculation (Exactly the same as your HRNet code)
    if len(stats) > 0:
        tp, conf, pcls, gcls = [torch.cat(x, 0).numpy() for x in zip(*stats)]
        results = ap_per_class(tp, conf, pcls, gcls, plot=False)
        
        # results structure: (tp, fp, p, r, f1, ap, unique_classes)
        ap = results[5] 
        map50 = ap[:, 0].mean() 
        map50_95 = ap.mean()
        
        return len(stats), results[2].mean(), results[3].mean(), map50, map50_95
    else:
        return None

def run_test_on_yolo_format(pipeline, test_img_dir, test_label_dir, device='cuda'):
    all_stats = []
  
    SIGMAS =  torch.tensor([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079]).to(device) # first 7 sigma values from the COCO sigmas
    IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10).to(device) #COCO standard thresholds
    
    image_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(test_img_dir, img_name)
       
        label_path = os.path.join(test_label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
       
        #  Pipeline Prediction
       
        results = pipeline.predict(img_path)
        if not results:
            continue

        # Convert predictions to [M, 7, 3] tensor
        pred_list = []
        for p in results:
            if pipeline.model_type == 'vitpose':
                p_kpts = torch.from_numpy(p.pred_instances.keypoints[0]) # [7, 2]
                p_scores = torch.from_numpy(p.pred_instances.keypoint_scores[0]).unsqueeze(-1) # [7, 1]
                p_final = torch.cat([p_kpts, p_scores], dim=-1) # [7, 3]
            else:
                p_final = torch.from_numpy(p) 
            pred_list.append(p_final)
        
        pred_kpts = torch.stack(pred_list).to(device) # [M, 7, 3]
        # Average keypoint score as object confidence 
        obj_conf = pred_kpts[:, :, 2].mean(dim=1) 

        # Load Ground Truth
        if os.path.exists(label_path):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            with open(label_path, 'r') as f:
                gt_lines = f.readlines()
            
            gt_list = []
            area_list = []
            for line in gt_lines:
                data = list(map(float, line.split()))
                # Compute Area
                area = (data[3] * w) * (data[4] * h) #the yolo format dataset has normalized coordinates
                area_list.append(area)
                
                # Keypoints: [7, 3]
                gk = np.array(data[5:]).reshape(-1, 3)
                gk[:, 0] *= w
                gk[:, 1] *= h
                
                gt_list.append(torch.from_numpy(gk))

            if not gt_list: continue
            
            gt_kpts = torch.stack(gt_list).to(device) # [N, 7, 3]
            gt_areas = torch.tensor(area_list).to(device) # [N]

            # Matching 
            # Returns [M, 10] (True/False per threshold)
            tp_matrix = process_batch(
                pred_kpts, 
                gt_kpts, 
                gt_areas, 
                SIGMAS, 
                obj_conf, 
                IOU_THRESHOLDS
            )
            
          
            all_stats.append((
                tp_matrix.cpu(), 
                obj_conf.cpu(), 
                torch.zeros(len(obj_conf)), 
                torch.zeros(len(gt_areas))
            ))

    # Compute Final mAP
    if len(all_stats) > 0:
        tp, conf, pcls, gcls = [torch.cat(x, 0).numpy() for x in zip(*all_stats)]
        results = ap_per_class(tp, conf, pcls, gcls, plot=False)
        
        # Results: (tp, fp, p, r, f1, ap, unique_classes)
        ap = results[5] 
        print(f"\n--- {pipeline.model_type.upper()} FINAL EVAL ---")
        print(f"Precision:    {results[2].mean():.4f}")
        print(f"Recall:       {results[3].mean():.4f}")
        print(f"mAP@50:    {ap[:, 0].mean():.4f}")
        print(f"mAP@50-95: {ap.mean():.4f}")
        return ap.mean()
    
    return 0