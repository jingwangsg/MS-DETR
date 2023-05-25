import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.nn_utils import clones
from einops import einsum, rearrange, repeat, reduce
import numpy as np
from kn_util.nn_utils.layers import MLP
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm


class SegFormerXAttention(nn.Module):

    def __init__(self, d_model, num_head, sr_ratio=1, dropout=0.1) -> None:
        super().__init__()
        d_head = d_model // num_head
        self.t2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.t2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.do = nn.Dropout(dropout)

        if sr_ratio > 1.0:
            self.sr = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                padding=(sr_ratio - 1) // 2,
            )
        self.sr_ratio = sr_ratio

        self.d_head = d_head
        self.num_head = num_head

        self.ff_text = nn.Linear(d_model, d_model)
        self.ff_vid = nn.Linear(d_model, d_model)

        self.ln_txt = LayerNorm(d_model, 1e-12)
        self.ln_vid = LayerNorm(d_model, 1e-12)

    def get_attn_logits(self, feat_k, mask_k, feat_q, mask_q, proj):
        B, Lq, _ = feat_q.shape
        B, Lk, _ = feat_k.shape
        B, Lk = mask_k.shape
        B, Lq = mask_q.shape

        feat_q = rearrange(
            proj[0](feat_q),
            "b lq (h dh) -> b lq h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        feat_k = rearrange(
            proj[1](feat_k),
            "b lk (h dh) -> b lk h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        attn_logits = einsum(feat_q, feat_k, "b lq h dh, b lk h dh->b h lq lk")
        attn_mask = repeat(
            einsum(mask_q, mask_k, "b lq, b lk->b lq lk"),
            "b lq lk->b h lq lk",
            h=self.num_head,
        )
        attn_logits[~attn_mask] = -10000.0

        return attn_logits  # b lv h lq lk

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape
        B, Lv = vid_mask.shape
        B, Lt = txt_mask.shape

        vid_feat_ = vid_feat
        txt_feat_ = txt_feat

        if self.sr_ratio > 1.0:
            vid_feat_sr = vid_feat.transpose(1, 2)
            vid_feat_sr = self.sr(vid_feat_sr)
            vid_feat_sr = vid_feat_sr.transpose(1, 2)
        else:
            vid_feat_sr = vid_feat

        vid_mask_sr = vid_mask[:, None, :]
        vid_mask_sr = nn.MaxPool1d(kernel_size=self.sr_ratio, stride=self.sr_ratio)(vid_mask_sr.float())
        vid_mask_sr = (vid_mask_sr > 0)[:, 0, :]

        v2v_value = self.v2v_proj[2](vid_feat_sr)
        t2v_value = self.t2v_proj[2](txt_feat)
        v_value = torch.cat([v2v_value, t2v_value], dim=1)
        v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2t_value = self.v2t_proj[2](vid_feat_sr)
        t2t_value = self.t2t_proj[2](txt_feat)
        t_value = torch.cat([v2t_value, t2t_value], dim=1)
        t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

        v2v_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, vid_feat, vid_mask, self.v2v_proj)
        t2v_logits = self.get_attn_logits(txt_feat, txt_mask, vid_feat, vid_mask, self.t2v_proj)
        v2t_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, txt_feat, txt_mask, self.v2t_proj)
        t2t_logits = self.get_attn_logits(txt_feat, txt_mask, txt_feat, txt_mask, self.t2t_proj)

        v_logits = torch.cat([v2v_logits, t2v_logits], dim=-1)
        v_logits = self.do(v_logits)
        v_logits /= np.sqrt(self.d_head)
        t_logits = torch.cat([v2t_logits, t2t_logits], dim=-1)
        t_logits = self.do(t_logits)
        t_logits /= np.sqrt(self.d_head)

        vid_feat = einsum(
            F.softmax(v_logits, dim=-1),
            v_value,
            "b h lq lk, b lk h d -> b lq h d",
        )
        vid_feat = rearrange(vid_feat, "b lq h d -> b lq (h d)")

        txt_feat = einsum(
            F.softmax(t_logits, dim=-1),
            t_value,
            "b h lq lk, b lk h d -> b lq h d",
        )
        txt_feat = rearrange(txt_feat, "b lq h d -> b lq (h d)")

        txt_feat = self.do(self.ff_text(txt_feat))
        vid_feat = self.do(self.ff_vid(vid_feat))

        txt_feat = self.ln_txt(txt_feat + txt_feat_)
        vid_feat = self.ln_vid(vid_feat + vid_feat_)

        return vid_feat, txt_feat


class SegFormerXEncoderLayer(nn.Module):

    def __init__(self, d_model, num_head, ff_dim, sr_ratio, dropout) -> None:
        super().__init__()
        self.cross_attn = SegFormerXAttention(d_model=d_model, num_head=num_head, sr_ratio=sr_ratio, dropout=dropout)
        self.ff_txt = MLP([d_model, ff_dim, d_model], activation="gelu")
        self.ff_vid = MLP([d_model, ff_dim, d_model], activation="gelu")
        self.ln_txt = LayerNorm(d_model, eps=1e-12)
        self.ln_vid = LayerNorm(d_model, eps=1e-12)
        self.do = nn.Dropout(dropout)

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask):
        """
        x = temporal_attn(x) + x
        x = cross_attn(x) + x
        x = OUTPUT(x)
        """
        B, Lv, _ = vid_feat.shape

        vid_feat, txt_feat = self.cross_attn(vid_feat, vid_mask, txt_feat, txt_mask)

        vid_feat_ = vid_feat
        txt_feat_ = txt_feat

        vid_feat = self.do(self.ff_vid(vid_feat))
        vid_feat = self.ln_vid(vid_feat + vid_feat_)

        txt_feat = self.do(self.ff_txt(txt_feat))
        txt_feat = self.ln_txt(txt_feat + txt_feat_)

        return vid_feat, txt_feat


class SegFormerXEncoder(nn.Module):

    def __init__(self, d_model_in, d_model_lvls, num_head_lvls, sr_ratio_lvls, ff_dim_lvls, use_patch_merge,
                 dropout) -> None:
        super().__init__()
        assert (len(d_model_lvls) == len(num_head_lvls) == len(sr_ratio_lvls) == len(ff_dim_lvls))
        self.layers = nn.ModuleList([
            SegFormerXEncoderLayer(d_model=d_model,
                                   num_head=num_head,
                                   ff_dim=ff_dim,
                                   sr_ratio=sr_ratio,
                                   dropout=dropout)
            for d_model, num_head, sr_ratio, ff_dim in zip(d_model_lvls, num_head_lvls, sr_ratio_lvls, ff_dim_lvls)
        ])

        d_model_lvls_ = [d_model_in] + d_model_lvls
        self.pe_lns = nn.ModuleList([nn.LayerNorm(d_model_lvls[i], 1e-12) for i in range(len(d_model_lvls))])
        self.txt_lvl_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model_lvls_[i - 1], d_model_lvls_[i]),
                LayerNorm(d_model_lvls_[i], eps=1e-12),
            ) for i in range(1, len(d_model_lvls_))
        ])
        self.use_patch_merge = use_patch_merge
        self.patch_merge = nn.ModuleList([
            nn.Conv1d(in_channels=d_model_lvls_[i - 1],
                      out_channels=d_model_lvls_[i],
                      kernel_size=3,
                      stride=2,
                      padding=1) for i in range(1, len(d_model_lvls_))
        ])

    @staticmethod
    def _interpolate_to_same_size(x, size):
        # B, L, D = x.shape
        x = x.transpose(1, 2)
        # x = F.interpolate(x, size=size, mode="linear")
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.transpose(1, 2)
        return x

    def _patch_merge(self, vid_feat, idx):
        vid_feat = vid_feat.transpose(1, 2)
        vid_feat = self.patch_merge[idx](vid_feat)
        vid_feat = vid_feat.transpose(1, 2)
        return vid_feat

    def forward(self, vid_feat, vid_mask, txt_feat, txt_mask, vid_pe=None):
        # vid_pe not None -> add vid_pe to each level of vid_feat
        B, Lv, _ = vid_feat.shape
        intermediate_states = []
        intermediate_masks = []
        for idx, layer in enumerate(self.layers):
            if self.use_patch_merge[idx]:
                vid_feat = self._patch_merge(vid_feat, idx=idx)
                vid_mask = self._interpolate_to_same_size(vid_mask[:, :, None].float(),
                                                          vid_feat.shape[1]).squeeze(-1).bool()
                if vid_pe is not None:
                    vid_pe = self._interpolate_to_same_size(vid_pe, vid_feat.shape[1])
                    vid_feat = self.pe_lns[idx](vid_pe + vid_feat)

            intermediate_states += [vid_feat]
            intermediate_masks += [vid_mask]
            txt_feat = self.txt_lvl_projs[idx](txt_feat)

            vid_feat, txt_feat = layer(vid_feat, vid_mask, txt_feat, txt_mask)

        return intermediate_states, intermediate_masks


class SegFormerX(nn.Module):
    """Video-text version of Segformer"""

    def __init__(self,
                 d_model_in=128,
                 d_model_lvls=[128, 256, 512, 1024],
                 num_head_lvls=[2, 4, 8, 16],
                 ff_dim_lvls=[256, 512, 1024, 2048],
                 sr_ratio_lvls=[8, 4, 2, 1],
                 input_vid_dim=768,
                 input_txt_dim=768,
                 max_vid_len=256,
                 max_txt_len=30,
                 dropout=0.1,
                 pe="learn",
                 pe_kernel_size=3,
                 use_patch_merge=[True, False, True, False],
                 output_layers=None) -> None:
        super().__init__()

        self.vid_proj = nn.Linear(input_vid_dim, d_model_in)
        self.txt_proj = nn.Linear(input_txt_dim, d_model_in)
        self.mask_embedding = nn.Embedding(1, d_model_in)

        self.pe = pe
        if pe == "learn":
            self.vid_pe = nn.Embedding(max_vid_len, d_model_in)
        elif pe == "conv":
            self.conv_pe = nn.Conv1d(in_channels=d_model_in,
                                     out_channels=d_model_in,
                                     kernel_size=pe_kernel_size,
                                     padding=(pe_kernel_size - 1) // 2)
        else:
            raise NotImplementedError()
        self.txt_pe = nn.Embedding(max_txt_len, d_model_in)
        # self.type_embed_vid = nn.Parameter(torch.randn(d_model_in))
        # self.type_embed_vid.data.normal_(mean=0.0, std=0.01)
        # self.type_embed_txt = nn.Parameter(torch.randn(d_model_in))
        # self.type_embed_txt.data.normal_(mean=0.0, std=0.02)
        self.vid_ln = LayerNorm(d_model_in, eps=1e-12)
        self.txt_ln = LayerNorm(d_model_in, eps=1e-12)
        self.do = nn.Dropout(dropout)

        self.encoder = SegFormerXEncoder(d_model_in=d_model_in,
                                         d_model_lvls=d_model_lvls,
                                         num_head_lvls=num_head_lvls,
                                         sr_ratio_lvls=sr_ratio_lvls,
                                         ff_dim_lvls=ff_dim_lvls,
                                         dropout=dropout,
                                         use_patch_merge=use_patch_merge)
        self.output_layers = [_ for _ in range(len(sr_ratio_lvls))] if output_layers is None else output_layers

        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_embedding(self, vid_feat, txt_feat, word_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape

        vid_feat = self.vid_proj(vid_feat)
        txt_feat = self.txt_proj(txt_feat)

        # if self.pe == "learn":
        vid_pe = self.vid_pe.weight[None, :vid_feat.shape[1]]
        # elif self.pe == "conv":
        #     pe = self.conv_pe(vid_feat.transpose(1, 2))
        #     pe = pe.transpose(1, 2)
        vid_feat = self.vid_ln(vid_feat + vid_pe)
        vid_feat = self.do(vid_feat)

        txt_pe = self.txt_pe.weight[None, :txt_feat.shape[1]]

        if self.training:
            _zero_idxs = torch.zeros(txt_feat.shape[:2], device=vid_feat.device, dtype=torch.long)
            txt_feat[word_mask] = self.mask_embedding(_zero_idxs)[word_mask].to(txt_feat.dtype)  # make AMP happy

        txt_feat = self.txt_ln(txt_feat + txt_pe)
        txt_feat = self.do(txt_feat)

        return vid_feat, txt_feat, vid_pe

    def forward(self, txt_feat, txt_mask, vid_feat, word_mask):
        B, Lv, _ = vid_feat.shape
        B, Lt, _ = txt_feat.shape
        txt_mask = txt_mask.bool()

        vid_feat, txt_feat, vid_pe = self._get_embedding(vid_feat, txt_feat, word_mask)
        vid_mask = torch.ones(vid_feat.shape[:2], dtype=torch.bool, device=txt_feat.device)
        intermediate_states, intermediate_masks = self.encoder(vid_feat, vid_mask, txt_feat, txt_mask, vid_pe)

        selected_interm_state = [intermediate_states[i] for i in self.output_layers]

        return selected_interm_state, txt_feat


class SegFormerXFPN(nn.Module):

    def __init__(self,
                 backbone,
                 output_layer=[0, 2, 3],
                 intermediate_hidden_size=[512, 512, 512],
                 fpn_hidden_size=512) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_layer = output_layer
        self.adapters = nn.ModuleList([
            nn.Conv1d(in_channels=cur_size, out_channels=fpn_hidden_size, kernel_size=1)
            for cur_size in intermediate_hidden_size
        ])  # adapt to a fixed fpn_hidden_size
        self.out_convs = nn.ModuleList([
            nn.Conv1d(in_channels=fpn_hidden_size, out_channels=fpn_hidden_size, kernel_size=3, padding=1)
            for _ in intermediate_hidden_size
        ])

    def forward(self, vid_feat, txt_feat, txt_mask, word_mask):
        intermediate_states, txt_feat = self.backbone(vid_feat=vid_feat,
                                                      txt_feat=txt_feat,
                                                      txt_mask=txt_mask,
                                                      word_mask=word_mask)
        fpn_states = []

        for idx, adapter in enumerate(self.adapters):
            state_idx = self.output_layer[idx]
            cur_state = intermediate_states[state_idx].transpose(1, 2)
            fpn_states += [adapter(cur_state)]

        for idx in range(0, len(self.adapters) - 1):
            fpn_states[idx] += F.interpolate(fpn_states[idx + 1], fpn_states[idx].shape[-1])

        for idx, fpn_state in enumerate(fpn_states):
            fpn_states[idx] = self.out_convs[idx](fpn_state).transpose(1, 2)

        return fpn_states, txt_feat