import torch
from torch import nn
from .frame_modules import FrameAvgPool
from .bert_modules import TLocVLBERT
from .loss import bce_rescale_loss
from misc import nms
from einops import rearrange, repeat
from kn_util.config import instantiate
import os
from kn_util.debug import sync_diff


class TAN(nn.Module):

    def __init__(self, frame_layer, bert_layer, cfg):
        super(TAN, self).__init__()

        self.frame_layer = frame_layer
        self.bert_layer = bert_layer
        self.cfg = cfg

    def forward_logits(self, textual_input, textual_mask, word_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)

        logits_text, logits_visual, logits_iou, iou_mask_map = self.bert_layer(textual_input, textual_mask, word_mask,
                                                                               vis_h)
        # logits_text = logits_text.transpose(1, 2)
        logits_visual = logits_visual.transpose(1, 2)

        return logits_text, logits_visual, logits_iou, iou_mask_map

    def forward(self,
                textual_input,
                textual_mask,
                visual_input,
                gt_maps,
                gt_times,
                word_label=None,
                word_mask=None,
                mode="train",
                **kwargs):
        word_mask = torch.zeros_like(textual_mask, dtype=torch.float, device=textual_input.device) \
            if word_mask is None else word_mask.float()
        word_label = torch.ones(textual_mask.shape, dtype=torch.long, device=textual_mask.device) \
            if word_label is None else word_label

        logits_text, logits_visual, logits_iou, iou_mask_map = self.forward_logits(textual_input, textual_mask,
                                                                                   word_mask, visual_input)

        losses, joint_prob, score, s_e_time = bce_rescale_loss(config=self.cfg,
                                                               logits_text=logits_text,
                                                               logits_visual=logits_visual,
                                                               logits_iou=logits_iou,
                                                               iou_mask_map=iou_mask_map,
                                                               gt_maps=gt_maps,
                                                               gt_times=gt_times,
                                                               word_label=word_label,
                                                               word_mask=word_mask)

        if mode == "train":
            return losses
        if mode == "test":
            B, ST, ED = score.shape
            score_batch = rearrange(score, "b st ed -> b (st ed)")
            s_e_time_batch = rearrange(s_e_time, "b i st ed -> b (st ed) i")
            batch_indices = repeat(torch.arange(B, device=textual_input.device), "b -> b i", i=ST * ED)
            valid_indices = (score_batch > 0) & (s_e_time_batch[..., 0] < s_e_time_batch[..., 1])
            score_valid = score_batch[valid_indices]
            s_e_time_valid = s_e_time_batch[valid_indices]
            batch_indices_valid = batch_indices[valid_indices]
            nms_s_e_time, nms_score = nms(s_e_time_valid,
                                          score_valid,
                                          batch_indices_valid,
                                          iou_threshold=self.cfg.nms_threshold)
            num_clips = self.cfg.num_clips
            boxxes = [_ / num_clips for _ in nms_s_e_time]

            return dict(boxxes=boxxes, s_e_time=nms_s_e_time, scores=nms_score, gt=gt_times / num_clips, **losses)
