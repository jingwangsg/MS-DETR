import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange, repeat
from detectron2.config import instantiate
from misc import calc_iou
from torchvision.ops import sigmoid_focal_loss

def l1_loss(pred_bds, gt):
    return (pred_bds - gt[:, None, :]).abs().mean()

def iou_loss(iou_scores, pred_bds, gt, alpha=2, iou_type="iou"):
    loss = 0.0
    # if isinstance(pred_bds, torch.Tensor):
    B, Nc, _2 = pred_bds.shape
    pred_bds_flatten = rearrange(pred_bds, "b nc i -> (b nc) i")
    gt_flatten = repeat(gt, "b i -> (b nc) i", nc=Nc)
    iou_gt_flatten = calc_iou(pred_bds_flatten, gt_flatten, type=iou_type)
    
    iou_scores_flatten = rearrange(iou_scores, "b nc-> (b nc)")
    loss = sigmoid_focal_loss(iou_scores_flatten, iou_gt_flatten, reduction="mean")
    return loss