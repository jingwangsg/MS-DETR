# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from ..ms_temporal_detr.ms_pooler import MultiScaleRoIAlign1D
# from misc import calc_iou
# from torchvision.ops import sigmoid_focal_loss
# from kn_util.nn_utils.layers import MLP

# def ms_deform_attn_pytorch(value, spatial_shapes, sampling_location):
#     value = value.split(spatial_shapes, )

# class MSDeformableAttention(nn.Module):
#     def __init__(self, d_model, n_levels=3, n_head=8, n_point=16) -> None:
#         super().__init__()
#         self.n_head = n_head
#         self.n_point = n_point
#         self.n_levels = n_levels

#         self.w_value = nn.Linear(d_model, d_model)
#         self.w_output = nn.Linear(d_model, d_model)
#         self.w_offset = nn.Linear(d_model, n_levels * n_head * n_point)
#         self.w_attn_weight = nn.Linear(d_model, n_levels * n_head * n_point)
    
#     def _init_weight(self):
#         pass
    
#     def forward(self, vid_feat_lvls, reference_points):
#         # B, (Lv_0 + ... + Lv_N), _ = vid_feat_lvls.shape
#         B, Nlvl, Nh = reference_points.shape
#         vid_feat_lvls = self.w_value(vid_feat_lvls)
#         attention_score = self.w_attn_weight(vid_feat_lvls).
#         attention_score = F.softmax(attention_score, dim=)
        

# class DeformableDecoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.attn = DeformableAttention()
    
#     def forward(self, vid_feat_lvls, reference_points):
#         pass
        

# class MultiScaleTemporalDetr(nn.Module):

#     def __init__(self, backbone, head, model_cfg) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
#         self.model_cfg = model_cfg

#         self.binary_ffn = MLP(model_cfg.d_model, model_cfg.d_model, 1)
    
#     def compute_binary_loss(self, binary_logits, map_gt):
#         binary_loss = sigmoid_focal_loss(binary_logits, map_gt, alpha=0.5)
#         return binary_loss.mean()
    
#     def compute_loss(self):
#         pass

#     def forward(self, vid_feat, txt_feat, txt_mask, gt=None, mode="train", **kwargs):
#         if word_mask is None and word_label is None:
#             word_mask = torch.zeros(txt_mask.shape, dtype=torch.bool, device=vid_feat.device)
#             word_label = torch.zeros(txt_mask.shape, dtype=torch.long, device=vid_feat.device)
#         vid_feat_lvls, txt_feat = self.backbone(vid_feat=vid_feat,
#                                                 txt_feat=txt_feat,
#                                                 txt_mask=txt_mask,
#                                                 word_mask=word_mask)
#         binary_logits = self.binary_ffn(vid_feat_lvls[-1])
#         topk_indices = torch.topk(binary_logits, k=self.model_cfg.num_query, dim=1)
#         num_clips = self.model_cfg.num_clips
#         reference_centers = topk_indices / num_clips

#         self.head(vid_feat_lvls, reference_centers)
#         losses = self.compute_loss()
#         if mode == "train":
#             return losses
#         # else:
#         #     return dict(scores=score_lvls[-1], boxxes=proposal_lvls[-1], **losses)
