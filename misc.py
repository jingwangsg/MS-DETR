import torch
from torchvision.ops import batched_nms
from einops import repeat, rearrange
import numpy as np
import torch.nn.functional as F
import os.path as osp
from kn_util.basic import global_get, global_set


def format_str(v, decimals=4):
    if isinstance(v, (float, np.float_)):
        return f"{v:.4f}"
    else:
        return str(v)


def dict2str(cur_dict, ordered_keys=None, keep_unordered=True):
    if ordered_keys is None:
        return "\t".join([k + " " + format_str(v) for k, v in cur_dict.items()])
    else:
        ordered = [k + " " + format_str(cur_dict[k]) for k in ordered_keys]
        if keep_unordered:
            unordered = [k + " " + format_str(v) for k, v in cur_dict.items()]
            outputs = ordered
        else:
            outputs = ordered
        return "\t".join(outputs)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def nms(pred_bds, scores, batch_idxs, iou_threshold):
    B, _2 = pred_bds.shape

    zero_pad = torch.zeros(pred_bds.shape[:1], dtype=torch.float32, device=pred_bds.device)
    one_pad = zero_pad + 1
    boxxes = torch.stack([pred_bds[:, 0], zero_pad, pred_bds[:, 1], one_pad], dim=-1)
    boxxes_flatten = boxxes
    scores_flatten = scores

    nms_indices = batched_nms(boxxes_flatten, scores_flatten, batch_idxs, iou_threshold)
    nms_pred_bds_flatten = boxxes_flatten[nms_indices][:, (0, 2)]
    nms_scores_flatten = scores_flatten[nms_indices]
    nms_idxs = batch_idxs[nms_indices]

    nms_pred_bds = []
    nms_scores = []
    for b in range(torch.max(batch_idxs).item() + 1):
        cur_batch_indices = (nms_idxs == b)
        nms_pred_bds.append(nms_pred_bds_flatten[cur_batch_indices])
        nms_scores.append(nms_scores_flatten[cur_batch_indices])

    return nms_pred_bds, nms_scores


@torch.no_grad()
def calc_iou(pred_bds, gt, type="iou"):
    """make sure the range between [0, 1) to make loss function happy"""
    min_ed = torch.minimum(pred_bds[:, 1], gt[:, 1])
    max_ed = torch.maximum(pred_bds[:, 1], gt[:, 1])
    min_st = torch.minimum(pred_bds[:, 0], gt[:, 0])
    max_st = torch.maximum(pred_bds[:, 0], gt[:, 0])

    I = torch.maximum(min_ed - max_st, torch.zeros_like(min_ed, dtype=torch.float, device=pred_bds.device))
    area_pred = pred_bds[:, 1] - pred_bds[:, 0]
    area_gt = gt[:, 1] - gt[:, 0]
    U = area_pred + area_gt - I
    Ac = max_ed - min_st

    iou = I / U

    if type == "iou":
        return iou
    elif type == "giou":
        return 0.5 * (iou + U / Ac)
    else:
        raise NotImplementedError()


def grid_sample1d(input, grid, padding_mode="zeros", align_corners=True):
    shape = grid.shape
    input = input.unsqueeze(-1)  # batch_size * C * L_in * 1
    grid = grid.unsqueeze(1)  # batch_size * 1 * L_out
    grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
    z = F.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    C = input.shape[1]
    out_shape = [shape[0], C, shape[1]]
    z = z.view(*out_shape)  # batch_size * C * L_out
    return z


def cw2se(cw, fix_out_of_bound=False):
    se = torch.zeros_like(cw)
    se[..., 0] = cw[..., 0] - cw[..., 1] / 2
    se[..., 1] = cw[..., 0] + cw[..., 1] / 2
    if fix_out_of_bound:
        se[..., 0][se[..., 0] < 0.0] = 0.0
        se[..., 1][se[..., 1] > 1.0] = 1.0

    # se[(se[..., 0] < 0.0) | (se[..., 1] > 1.0)] = 0.0

    return se

from torchdata.datapipes.iter import IterDataPipe

class WordLabelTranslater(IterDataPipe):
    def __init__(self, src_pipeline, from_key) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        cfg = global_get("cfg")
        self.cfg = cfg
        itos, vocab = global_get("glove_vocab")
        self.itos = itos

        vocab_file=osp.join(cfg.data.dataset_dir, "annot", "vocab.txt")
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        topk_vocab = [w.strip() for w in lines]
        global_set("vocab_size", len(topk_vocab))
        self.topk_itos = topk_vocab
        self.topk_stoi = {w: idx for idx, w in enumerate(self.topk_itos)}

        self.unk_ind = self.topk_stoi["UNK"]
        self.from_key = from_key
    
    def __iter__(self):
        for x in self.src_pipeline:
            word_label = x[self.from_key].tolist()
            text_tok = [self.itos[w] for w in word_label]
            converted_word_label = [self.topk_stoi.get(w, self.unk_ind) for w in text_tok]
            x[self.from_key + ".topk_freq"] = np.array(converted_word_label)
            yield x

            