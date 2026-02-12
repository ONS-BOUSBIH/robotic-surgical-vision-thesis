##### PROBABLY no use for this file anymore, to be deleted!!!######################

import torch
import torch.nn as nn
import numpy as np


# Custom Arg max prediction function: determines keypoints location from predicted heatmaps 
## but add a confidence threshold to assess if the heatmaps really predicts a location or not.
def custom_get_max_preds(batch_heatmaps, conf_threshold=0.1):
    
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
   
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = np.floor(preds[:, :, 1] / width)

    # Create mask based on confidence
    pred_mask = (maxvals > conf_threshold).astype(np.float32)
    
    pred_mask = np.tile(pred_mask, (1, 1, 2))

    preds *= pred_mask  # zero out invisible joints

    return preds, maxvals

#Custom Loss fonction to handle invisible and visible keypoints differently
class WeightedJointsMSELoss(nn.Module):
    def __init__(self, visible_weight=1.0, invisible_weight=0.2):
        super(WeightedJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.visible_weight = visible_weight
        self.invisible_weight = invisible_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0.0
        joint_weight = (
            target_weight * self.visible_weight +
            (1 - target_weight) * self.invisible_weight
        )

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            w = joint_weight[:, idx]
            loss += 0.5 * self.criterion(
                    heatmap_pred*w,
                    heatmap_gt*w)
         

        return loss / num_joints

## Inference performance metrics

def mpjpe(pred, gt, mask=None, mean=True):
    '''Mean Per Joint Position Error, gives the average of the Euclidian distance between
     predicted keypoints and ground-truth targets across all keypoints in all samples, 
    if a mask is provided, only valid keypoints are included
    in the calculation.
    pred, gt: (batch_size, number of joints, 2)
    mask: ((batch_size, number of joints)
    '''
    dist = np.linalg.norm(pred - gt, axis=2)  
    if mask is not None:
        dist = dist * mask  # ignore missing key points
        total_valid = np.sum(mask)
        total_valid = np.maximum(total_valid, 1e-6)
        return np.sum(dist) / total_valid
    if mean:
        return np.mean(dist)
    else:
        return np.mean(dist, axis=1)

def per_joint_mae(pred, gt, mask=None, mean= True):
    '''Per Joint mean position error, gives the average of the Euclidian distance between
     predicted keypoints and ground-truth targets across all samples for each keypoint type, 
    if a mask is provided, only valid keypoints are included
    in the calculation.
    pred, gt: (batch_size, number of joints, 2)
    mask: ((batch_size, number of joints)
    '''
    dist = np.linalg.norm(pred - gt, axis=2)
    if mask is not None:
        dist = dist * mask
        total_valid = np.sum(mask, axis=0)  # per joint
        total_valid = np.maximum(total_valid, 1e-6)
        return np.sum(dist, axis=0) / total_valid
    if mean:
        return np.mean(dist, axis=0)
    else:
        return dist



def pck(pred, gt, thresh_px, mask=None):
    '''Perctentage of Correct Keypoints, gives the fraction of correctly predicted keypoints whose Euclidian distance to the targets keypoint is within 
    a chosen pixel threshold.
    if a mask is provided, only valid keypoints are included
    in the calculation.
    pred, gt: (batch_size, number of joints, 2)
    mask: ((batch_size, number of joints)
    thresh_px: int, pixel radius.

    '''
    dist = np.linalg.norm(pred - gt, axis=2)  # (N, K)
    correct = (dist <= thresh_px).astype(np.float32)  # (N, K)

    if mask is not None:
      
        correct = correct * mask
        total_valid = np.sum(mask)
        return np.sum(correct) / total_valid

    return np.mean(correct)


def visibility_counts(pred_mask, gt_mask):
    """
    gives the number of TP, FP, FN and TN predicted keypoints, considering the visibility criterion.
    - TP: keypoint is visible in both GT and prediction.
    - FP: keypoint is predicted visible but is not visible in GT.
    - FN: keypoint is visible in GT but predicted invisible.
    - TN: keypoint is invisible in both GT and prediction
    pred_mask, gt_mask: arrays of shape (batch, num_joints), values 0 or 1
    returns: TP, FP, FN, TN (ints)
    """
    pred = (pred_mask == 1)
    gt = (gt_mask == 1)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & (~gt))
    fn = np.sum((~pred) & gt)
    tn = np.sum((~pred) & (~gt))
    return int(tp), int(fp), int(fn), int(tn)


def visibility_precision(pred_mask, gt_mask, eps=1e-9):
    
    tp, fp, _, _ = visibility_counts(pred_mask, gt_mask)
    return tp / (tp + fp + eps)


def visibility_recall(pred_mask, gt_mask, eps=1e-9):
    tp, _, fn, _ = visibility_counts(pred_mask, gt_mask)
    return tp / (tp + fn + eps)


def visibility_f1(pred_mask, gt_mask, eps=1e-9):
    p = visibility_precision(pred_mask, gt_mask, eps)
    r = visibility_recall(pred_mask, gt_mask, eps)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def detection_rate(pred_conf, thresh, mask_gt):
    mask_pred = (pred_conf >= thresh).astype(np.float32)
    return np.sum(mask_pred * mask_gt) / (np.sum(mask_gt)), mask_pred



