# ============================================================================ #
# * use reference centers from output of encoder as the initial referecence of decoder querys
# ============================================================================ #

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
from kn_util.basic import registry
from ..backbone.segformerx import SegFormerXFPN, SegFormerX
from .ms_pooler import MultiScaleRoIAlign1D
from misc import cw2se, calc_iou
from ..loss import l1_loss, iou_loss
from kn_util.nn_utils.layers import MLP
from kn_util.nn_utils import clones
from kn_util.nn_utils.math import inverse_sigmoid_torch, gaussian_torch
from kn_util.basic import registry, global_get
from torchvision.ops import sigmoid_focal_loss
from kn_util.nn_utils.init import init_module
import copy
import os
from kn_util.basic import eval_env
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn import Linear, MultiheadAttention, Dropout, LayerNorm
from torch.nn.modules.transformer import _get_activation_fn
from kn_util.basic import eval_env
import os.path as osp

GROUP_IDX = eval_env("KN_DENOISING", 1)
KN_QUERY_ATTN = eval_env("KN_QUERY_ATTN", False)
KN_QUERY_ATTN_NOSHARE = eval_env("KN_QUERY_ATTN_NOSHARE", False)
KN_P_ADD_T = eval_env("KN_P_ADD_T", False)
local_dict = copy.copy(locals())
print({k: v for k, v in local_dict.items() if k.startswith("KN")})

_DEBUG_DIR = '/export/home2/kningtg/WORKSPACE/moment-retrieval-v2/query-moment/_debug/'


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn = self.multihead_attn(x,
                                      mem,
                                      mem,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask,
                                      need_weights=True)

        if eval_env("KN_DEBUG"):
            torch.save(attn, osp.join(_DEBUG_DIR, "attn.pt"))

        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class QueryBasedDecoder(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 ff_dim,
                 num_query,
                 num_layers=4,
                 num_scales=4,
                 pooler_resolution=16,
                 dim_init_ref=1,
                 dropout=0.1) -> None:
        super().__init__()

        # if os.getenv("KN_QUERY_GROUP", False):
        # query_group = global_get("cfg").model_cfg.query_group
        self.query_embeddings = clones(nn.Embedding(num_query, d_model), 5)
        # else:
        #     self.query_embeddings = nn.Embedding(num_query, d_model)
        self.d_model = d_model

        bbox_head = MLP([d_model, d_model, 2])
        self.bbox_heads = clones(bbox_head, num_layers)
        self.reference_head = MLP([d_model, d_model, dim_init_ref])
        score_head = MLP([d_model, d_model, 1])
        self.score_heads = clones(score_head, num_layers)

        vocab = global_get("glove_vocab")
        vocab_size = 1513 if vocab is None else len(vocab[0])

        self.num_layers = num_layers
        # make sure first offset is reasonable

        layer = TransformerDecoderLayer(d_model, nhead, ff_dim, dropout=dropout, batch_first=True)
        self.layers = clones(layer, num_layers)

        # build pooler
        self.pooler = MultiScaleRoIAlign1D(output_size=pooler_resolution)

        # prepare for processing pooled_feat
        if KN_QUERY_ATTN:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        elif KN_QUERY_ATTN_NOSHARE:
            self.attn0 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.w1 = nn.Linear(3 * d_model, d_model)
        else:
            pool_ffn = MLP([d_model * pooler_resolution * num_scales,\
                            d_model, d_model])
            self.pool_ffns = clones(pool_ffn, num_layers)

        self.init_weight()

    def init_weight(self):
        self.apply(init_module)
        for bbox_head in self.bbox_heads:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

        nn.init.constant_(self.bbox_heads[0].layers[-1].bias[1:], -2.0)
        # nn.init.normal_(self.bbox_heads[0], mean=-2, std=1)

    def get_initital_reference(self, offset, reference_no_sigmoid):
        # assume temporal length of first prediction totally depends on offset predicted by boxx_head
        offset[..., :1] += reference_no_sigmoid.unsqueeze(-1)

        return offset

    def forward(self, feat_lvls, reference_centers):
        B = feat_lvls[0].shape[0]

        proposal_lvls_group = []
        score_lvls_group = []

        for group_idx in range(GROUP_IDX):
            query_embeddings = self.query_embeddings[group_idx].weight
            memory = feat_lvls[-1]
            tgt = repeat(query_embeddings, "nq d -> b nq d", b=B)
            reference = None
            proposal_lvls = []
            score_lvls = []
            stage_logits_lvls = []

            for idx, layer in enumerate(self.layers):
                output = layer(tgt, memory)  # B, Nq, D

                # box score
                score_logits = self.score_heads[idx](output)
                score = score_logits.squeeze(-1)
                score_lvls.append(score)

                # box refine / get initial box
                offset = self.bbox_heads[idx](output)
                reference_no_sigmoid = inverse_sigmoid_torch(reference_centers, eps=1e-12)
                if idx == 0:
                    offset = self.get_initital_reference(offset, reference_no_sigmoid)
                else:
                    offset = inverse_sigmoid_torch(reference) + offset
                reference = offset.sigmoid()
                proposal = cw2se(reference, fix_out_of_bound=not self.training)
                proposal_lvls.append(proposal)

                proposal = proposal.detach()

                # roi pool & concat
                pooled_feat_list = self.pooler(feat_lvls, proposal)
                # [[(Nq,Rp,D)...x B]...x N_lvls]
                pooled_feat_list = [torch.stack(x, dim=0) for x in pooled_feat_list]
                pooled_feat = torch.stack(pooled_feat_list, dim=0)
                # B, N_lvls, Nq, Rp, D

                if KN_QUERY_ATTN:
                    pooled_feat = rearrange(pooled_feat, "nlvl b nq rd d -> b nq (nlvl rd) d")
                    expanded_query_embeddings = repeat(query_embeddings, "nq d -> (b nq) d", b=B)
                    pooled_feat = rearrange(pooled_feat, "b nq p d -> (b nq) p d")
                    pooled_feat = self.attn(key=pooled_feat,
                                            query=expanded_query_embeddings[:, None],
                                            value=pooled_feat)[0]
                    pooled_feat = rearrange(pooled_feat, "(b nq) i d -> b nq (i d)", b=B)
                elif KN_QUERY_ATTN_NOSHARE:
                    expanded_query_embeddings = repeat(query_embeddings, "nq d -> (b nq) d", b=B)
                    pooled_feat = rearrange(pooled_feat, "nlvl b nq rd d -> nlvl (b nq) rd d")
                    pooled_feat0 = self.attn0(key=pooled_feat[0],
                                              query=expanded_query_embeddings[:, None],
                                              value=pooled_feat[0])[0]
                    pooled_feat1 = self.attn0(key=pooled_feat[1],
                                              query=expanded_query_embeddings[:, None],
                                              value=pooled_feat[1])[0]
                    pooled_feat2 = self.attn0(key=pooled_feat[2],
                                              query=expanded_query_embeddings[:, None],
                                              value=pooled_feat[2])[0]  # (b nq) 1 d
                    pooled_feat = torch.cat([pooled_feat0, pooled_feat1, pooled_feat2], dim=-1)
                    pooled_feat = rearrange(pooled_feat, "(b nq) i d -> b nq (i d)", b=B)
                    pooled_feat = self.w1(pooled_feat)
                else:
                    pooled_feat = rearrange(pooled_feat, "nlvl b nq rd d -> b nq (nlvl rd d)")
                    pooled_feat = self.pool_ffns[idx](pooled_feat)

                if KN_P_ADD_T:
                    tgt = pooled_feat + tgt
                else:
                    tgt = pooled_feat + query_embeddings

            proposal_lvls_group.append(proposal_lvls)
            score_lvls_group.append(score_lvls)

            if not self.training:
                break  # only use first group

        return proposal_lvls_group, score_lvls_group


class MultiScaleTemporalDetr(nn.Module):

    def __init__(self, backbone, head, frame_pooler, model_cfg) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.model_cfg = model_cfg
        self.frame_pooler = frame_pooler

        d_model = model_cfg.d_model

        # stage mlp
        self.stage_mlp1 = nn.Linear(d_model, 3 * d_model)
        self.stage_mlps = clones(MLP([d_model, d_model, 1]), 3)

        vocab_size = global_get("vocab_size")
        self.text_mlp = MLP([d_model, d_model, vocab_size])

        self.stage_mlp1.apply(init_module)
        self.stage_mlps.apply(init_module)
        self.text_mlp.apply(init_module)

    def compute_boudary_loss(self, proposal, score, gt, idx):
        cfg = self.model_cfg
        loss = 0.0

        B, Nc, _2 = proposal.shape

        iou_gt_flatten = calc_iou(proposal.reshape(B * Nc, 2),
                                  gt[:, None].repeat(1, Nc, 1).reshape(B * Nc, 2),
                                  type="iou")
        iou_gt = iou_gt_flatten.reshape((B, Nc))

        topk_indices_flatten = torch.topk(iou_gt, k=cfg.topk, dim=1).indices.flatten()
        batch_indices_flatten = torch.arange(0, B, device=proposal.device)[:, None].repeat(1, cfg.topk).flatten()
        # iou_gt[iou_gt < 0.5] = 0.0
        iou_gt[iou_gt < cfg.iou_cutoff] = 0.0
        iou_gt[batch_indices_flatten, topk_indices_flatten] = 1.0

        val_iou_loss = sigmoid_focal_loss(score.flatten(), iou_gt.flatten(), reduction="mean")
        loss += val_iou_loss * cfg.w_iou_loss

        topk_proposal = proposal[batch_indices_flatten, topk_indices_flatten].reshape(B, cfg.topk, 2)

        val_l1_loss = l1_loss(topk_proposal, gt)
        loss += val_l1_loss * cfg.w_l1_loss

        return dict(loss=loss, l1_loss=val_l1_loss, iou_loss=val_iou_loss, topk_indices_flatten=topk_indices_flatten)

    def compute_stage_loss(self, stage_logits, map_gt):
        map_gt = map_gt[:, :3, :].transpose(1, 2)
        stage_loss = sigmoid_focal_loss(stage_logits, map_gt, reduction="mean", alpha=0.5)
        return stage_loss

    def compute_iou_loss(self, proposal, score, gt):
        # calc_iou(gt)
        Nc = proposal.shape[1]
        expanded_gt = repeat(gt, "b i -> (b nc) i", nc=Nc)
        proposal = rearrange(proposal, "b nc i -> (b nc) i", nc=Nc)
        iou_type = "giou" if os.getenv("KN_SOFT_GIOU", False) else "iou"
        ious = calc_iou(proposal, expanded_gt, type=iou_type)
        score_flatten = score.flatten()
        return sigmoid_focal_loss(score_flatten, ious, alpha=0.5, reduction="mean")

    def compute_span_loss(self, stage_logits, gt):
        bsz = gt.shape[0]
        num_clips = self.model_cfg.num_clips
        map_gt = torch.zeros(bsz, num_clips, device=stage_logits.device, dtype=torch.float)
        gt_idxs = torch.round(gt * (num_clips - 1)).long()

        range_idxs = torch.arange(num_clips, device=stage_logits.device)
        map_gt = torch.logical_and(range_idxs[None,] >= gt_idxs[:, :1], range_idxs[None,] <= gt_idxs[:, 1:])
        map_gt = map_gt.float()

        return sigmoid_focal_loss(stage_logits[..., 0], map_gt, alpha=0.5, reduction="mean")

    def compute_loss(self, proposal_lvls_group, score_lvls_group, stage_logits, text_logits, word_mask, word_label, gt):
        cfg = self.model_cfg
        loss = 0.0
        loss_dict = dict()

        # decoder boundary loss
        all_bd_loss = []
        Ng = len(proposal_lvls_group)
        for group_idx in range(Ng):
            for idx, (proposal, score) in enumerate(zip(proposal_lvls_group[group_idx], score_lvls_group[group_idx])):
                bd_loss_dict = self.compute_boudary_loss(proposal, score, gt, idx=idx)
                bd_loss = bd_loss_dict["loss"]
                all_bd_loss.append(bd_loss)
                loss += bd_loss * cfg.w_bd_loss[idx] / Ng
                loss_dict[f"bd_{idx}_gr{group_idx}_loss"] = bd_loss

        # stage loss
        span_loss = self.compute_span_loss(stage_logits, gt)
        assert hasattr(cfg, "w_span_loss")
        loss += span_loss * cfg.w_span_loss
        loss_dict[f"span_loss"] = span_loss

        # text loss
        mask_loss = F.cross_entropy(text_logits.transpose(1, 2), word_label, reduction="none") * word_mask.float()
        mask_loss = mask_loss.mean()
        loss_dict["mask_loss"] = mask_loss
        loss += cfg.w_mask_loss * mask_loss

        loss_dict["loss"] = loss

        return loss_dict

    def forward(self, vid_feat, txt_feat, txt_mask, word_mask=None, word_label=None, gt=None, mode="train", **kwargs):
        B = vid_feat.shape[0]

        model_cfg = self.model_cfg
        vid_feat = self.frame_pooler(vid_feat)
        # word_mask = torch.zeros(txt_mask.shape, dtype=torch.bool, device=vid_feat.device)
        vid_feat_lvls, txt_feat = self.backbone(vid_feat=vid_feat,
                                                txt_feat=txt_feat,
                                                txt_mask=txt_mask,
                                                word_mask=word_mask)

        vid_feat_stage = self.stage_mlp1(vid_feat_lvls[-1])
        hidden_st, hidden_ed, hidden_md = torch.split(vid_feat_stage, model_cfg.d_model, dim=-1)
        logits_st = self.stage_mlps[0](hidden_st)
        logits_ed = self.stage_mlps[1](hidden_ed)
        logits_md = self.stage_mlps[2](hidden_md)
        stage_logits = torch.cat([logits_st, logits_ed, logits_md], dim=-1)

        topk_indices = torch.arange(0, model_cfg.num_query, device=vid_feat.device)
        topk_indices = repeat(topk_indices, "i -> b i", b=B)
        reference_centers = topk_indices / model_cfg.num_query
        reference_centers = reference_centers.detach()

        proposal_lvls_group, score_lvls_group = self.head(vid_feat_lvls, reference_centers)

        text_logits = self.text_mlp(txt_feat)

        if mode in ("train", "test"):
            losses = self.compute_loss(proposal_lvls_group, score_lvls_group, stage_logits, text_logits, word_mask,
                                       word_label, gt)
            if mode == "test":
                losses.update(dict(scores=score_lvls_group[0][-1], boxxes=proposal_lvls_group[0][-1]),
                              reference_centers=reference_centers,
                              stage_logits=stage_logits)
            return losses
        elif mode == "inference":
            return dict(scores=score_lvls_group[0][-1], boxxes=proposal_lvls_group[0][-1])
