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
import os


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
        self.query_embeddings = nn.Embedding(num_query, d_model)
        self.d_model = d_model

        bbox_head = MLP(d_model, d_model, 2, 3)
        self.bbox_heads = clones(bbox_head, num_layers)
        self.reference_head = MLP(d_model, d_model, dim_init_ref, 3)
        score_head = MLP(d_model, d_model, 1)
        self.score_heads = clones(score_head, num_layers)

        vocab = global_get("glove_vocab")
        vocab_size = 1513 if vocab is None else len(vocab[0])
        # text mlp
        self.text_clf = MLP(d_model, d_model, vocab_size)
        self.text_clf.apply(init_module)

        # stage mlp
        self.stage_mlp1 = clones(nn.Linear(d_model, 3 * d_model), num_scales)
        self.stage_mlps = clones(clones(MLP(d_model, d_model, 1), 3), num_scales)
        self.stage_mlp1.apply(init_module)
        self.stage_mlps.apply(init_module)

        self.num_layers = num_layers
        # make sure first offset is reasonable

        layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, dropout=dropout, batch_first=True)
        self.layers = clones(layer, num_layers)

        # build pooler
        self.pooler = MultiScaleRoIAlign1D(output_size=pooler_resolution)

        # prepare for processing pooled_feat
        pool_ffn = MLP(d_model * pooler_resolution * num_scales,\
                        d_model, d_model)
        self.pool_ffns = clones(pool_ffn, num_layers)
        self.pool_ffns.apply(init_module)

        nn.init.constant_(self.bbox_heads[0].layers[-1].bias.data[1:], -2.0)
        nn.init.constant_(bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_head.layers[-1].bias.data, 0)

    def get_initital_reference(self, offset, reference_no_sigmoid):
        if reference_no_sigmoid.shape[-1] == 1:
            # assume temporal length of first prediction totally depends on offset predicted by boxx_head
            offset[..., :1] += reference_no_sigmoid
        else:
            offset += reference_no_sigmoid

        return offset

    def forward(self, feat_lvls, txt_feat):
        B = feat_lvls[0].shape[0]
        text_logits = self.text_clf(txt_feat)

        query_embeddings = self.query_embeddings.weight
        memory = feat_lvls[-1]
        tgt = repeat(query_embeddings, "nq d -> b nq d", b=B)
        reference = None
        proposal_lvls = []
        score_lvls = []
        stage_logits_lvls = []

        for idx, vid_feat in enumerate(feat_lvls):
            vid_feat_stage = self.stage_mlp1[idx](vid_feat)
            hidden_st, hidden_ed, hidden_md = torch.split(vid_feat_stage, self.d_model, dim=-1)
            logits_st = self.stage_mlps[idx][0](hidden_st)
            logits_ed = self.stage_mlps[idx][1](hidden_ed)
            logits_md = self.stage_mlps[idx][2](hidden_md)
            stage_logits = torch.cat([logits_st, logits_ed, logits_md], dim=-1)
            stage_logits_lvls.append(stage_logits)

        for idx, layer in enumerate(self.layers):
            output = layer(tgt, memory)  # B, Nq, D

            # get score
            score_logits = self.score_heads[idx](output)
            score = score_logits.squeeze(-1)
            score_lvls.append(score)

            # box refine / get initial box
            offset = self.bbox_heads[idx](output)
            reference_no_sigmoid = self.reference_head(query_embeddings)
            if idx == 0:
                offset = self.get_initital_reference(offset, reference_no_sigmoid)
            else:
                offset = inverse_sigmoid_torch(reference) + offset
            reference = offset.sigmoid()
            proposal = cw2se(reference)
            proposal_lvls.append(proposal)

            # roi pool & concat
            pooled_feat_list = self.pooler(feat_lvls, proposal)
            # [[(Nq,Rp,D)...x B]...x N_lvls]
            pooled_feat_list = [torch.stack(x, dim=0) for x in pooled_feat_list]
            pooled_feat = torch.stack(pooled_feat_list, dim=0)
            # B, N_lvls, Nq, Rp, D
            pooled_feat = rearrange(pooled_feat, "nlvl b nq rd d -> b nq (nlvl rd d)")

            pooled_feat = self.pool_ffns[idx](pooled_feat)
            tgt = pooled_feat + query_embeddings

        return proposal_lvls, score_lvls, stage_logits_lvls, text_logits


class MultiScaleTemporalDetr(nn.Module):

    def __init__(self, backbone, head, model_cfg) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.model_cfg = model_cfg

    def compute_boudary_loss(self, proposal, score, gt, idx, topk_indices_flatten=None):
        cfg = self.model_cfg
        loss = 0.0

        from kn_util.basic import global_get
        logger = global_get("logger")
        # use_log = (idx == 2)
        use_log = False

        B, Nc, _2 = proposal.shape

        iou_gt_flatten = calc_iou(proposal.reshape(B * Nc, 2),
                                  gt[:, None].repeat(1, Nc, 1).reshape(B * Nc, 2),
                                  type="giou")
        iou_gt = iou_gt_flatten.reshape((B, Nc))

        if use_log:
            logger.info(f"======={idx}=======")
            logger.info("iou:")
            logger.info(iou_gt)

        if topk_indices_flatten is None:
            topk_indices_flatten = torch.topk(iou_gt, k=cfg.topk, dim=1).indices.flatten()
        batch_indices_flatten = torch.arange(0, B, device=proposal.device)[:, None].repeat(1, cfg.topk).flatten()

        if use_log:
            logger.info(f"topk_indices:")
            logger.info(topk_indices_flatten)

        # iou_gt[iou_gt < 0.5] = 0.0
        iou_gt[iou_gt < cfg.iou_cutoff] = 0.0
        iou_gt[batch_indices_flatten, topk_indices_flatten] = 1.0
        val_iou_loss = sigmoid_focal_loss(score.flatten(), iou_gt.flatten(), reduction="mean")
        loss += val_iou_loss * cfg.w_iou_loss

        topk_proposal = proposal[batch_indices_flatten, topk_indices_flatten].reshape(B, cfg.topk, 2)

        if use_log:
            logger.info("score:")
            logger.info(score.sigmoid())
            logger.info("topk_proposal:")
            logger.info(topk_proposal)
            logger.info("topk_score:")
            logger.info(score.sigmoid()[batch_indices_flatten, topk_indices_flatten])
            logger.info("gt:")
            logger.info(gt)

        val_l1_loss = l1_loss(topk_proposal, gt)
        loss += val_l1_loss * cfg.w_l1_loss

        return dict(loss=loss, l1_loss=val_l1_loss, iou_loss=val_iou_loss, topk_indices_flatten=topk_indices_flatten)

    def compute_stage_loss(self, stage_logits, gt):
        cfg = self.model_cfg
        sigma_se = 0.25
        sigma_m = 0.21

        B, Lv, _ = stage_logits.shape
        gt_mid = ((gt[:, 0] + gt[:, 1]) / 2)[:, None]
        gt_expanded = torch.cat([gt, gt_mid], dim=-1)
        gt_times = gt_expanded * Lv
        sigma = torch.tensor([sigma_se, sigma_se, sigma_m], device=gt.device)[None, :].repeat((B, 1))
        sigma = sigma * (gt_times[:, 1] - gt_times[:, 0])[:, None]

        stage_label = gaussian_torch(gt_times, sigma, Lv).transpose(1, 2)
        stage_label[stage_label > 0.8] = 1
        stage_label[stage_label < 0.4] = 0

        stage_loss = sigmoid_focal_loss(stage_logits, stage_label, reduction="mean")
        return stage_loss

    def compute_loss(self, proposal_lvls, score_lvls, text_logits, word_mask, word_label, stage_logits_lvls, gt):
        cfg = self.model_cfg
        loss = 0.0

        # l1 + iou loss
        final_layer_losses = self.compute_boudary_loss(proposal_lvls[-1],
                                                       score_lvls[-1],
                                                       gt,
                                                       idx=len(proposal_lvls) - 1)
        topk_indices_flatten = final_layer_losses.pop("topk_indices_flatten")
        loss += final_layer_losses["loss"]

        # decoder aux loss
        if cfg.get("cfg.w_dec_aux_loss", 0) > 0:
            dec_aux_loss = 0.0
            for idx, (proposal, score) in enumerate(zip(proposal_lvls[:-1], score_lvls[:-1])):
                topk = topk_indices_flatten if os.environ.get("KN_USE_SAME_TOPK", False) else None
                loss_dict = self.compute_boudary_loss(proposal, score, gt, idx=idx, topk_indices_flatten=topk)
                dec_aux_loss = loss_dict["loss"]
                loss += dec_aux_loss * cfg.w_dec_aux_loss

        # encoder aux loss
        if cfg.get("w_enc_aux_loss", 0) > 0:
            for stage_logits in stage_logits_lvls:
                enc_aux_loss = self.compute_stage_loss(stage_logits, gt)
                loss += enc_aux_loss * cfg.w_enc_aux_loss

        # mask loss
        _shape = text_logits.shape[:2]
        text_logits_flatten = text_logits.reshape((-1, text_logits.shape[-1]))
        word_label_flatten = word_label.flatten()
        mask_loss = F.cross_entropy(text_logits_flatten, word_label_flatten,
                                    reduction="none") * word_mask[..., None].float()
        mask_loss = mask_loss.mean()
        loss += mask_loss * cfg.w_mask_loss

        return dict(**final_layer_losses, mask_loss=mask_loss, enc_aux_loss=enc_aux_loss, dec_aux_loss=dec_aux_loss)

    def forward(self, vid_feat, txt_feat, txt_mask, word_mask=None, word_label=None, gt=None, mode="train", **kwargs):
        if word_mask is None and word_label is None:
            word_mask = torch.zeros(txt_mask.shape, dtype=torch.bool, device=vid_feat.device)
            word_label = torch.zeros(txt_mask.shape, dtype=torch.long, device=vid_feat.device)
        vid_feat_lvls, txt_feat = self.backbone(vid_feat=vid_feat,
                                                txt_feat=txt_feat,
                                                txt_mask=txt_mask,
                                                word_mask=word_mask)
        proposal_lvls, score_lvls, stage_logits, text_logits = self.head(vid_feat_lvls, txt_feat)

        losses = self.compute_loss(proposal_lvls, score_lvls, text_logits, word_mask, word_label, stage_logits, gt)
        if mode == "train":
            return losses
        else:
            return dict(scores=score_lvls[-1], boxxes=proposal_lvls[-1], **losses)
