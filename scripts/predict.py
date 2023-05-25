import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from kn_util.config import LazyConfig, instantiate
from kn_util.data import collection_to_device
from einops import repeat, rearrange
from tqdm import tqdm
from evaluater import ValTestEvaluater

from data.build import build_dataloader
from kn_util.basic import global_get, global_set, save_pickle
import numpy as np
from misc import calc_iou
from engine import evaluate

def post_process(out):
    ret = []
    topk = 5
    for idx in range(len(out["video_id"])):
        video_id = out["video_id"][idx]
        text = out["text"][idx]
        boxxes = out["boxxes"][idx].detach().cpu().numpy()
        scores = out["scores"][idx].sigmoid().detach().cpu().numpy()

        gt = out["gt"][idx].detach().cpu().numpy().tolist()
        attn = out["attn"][idx]
        ious = out["ious"][idx]

        indices = np.argsort(scores)[::-1]

        boxxes = boxxes[indices].tolist()
        scores = scores[indices].tolist()
        attn = attn[indices]
        ious = ious[indices]

        cur_item = dict(video_id=video_id, text=text, boxxes=boxxes, scores=scores, attn=attn, gt=gt, ious=ious)

        ret.append(cur_item)
    return ret


_DEBUG_DIR = '/export/home2/kningtg/WORKSPACE/moment-retrieval-v2/query-moment/_debug/'

exp_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval-v2/query-moment/work_dir/activitynet/mst_detr_v4-cross-pr16-train_sch10-group_lr-dec5"
ckpt_fn = glob.glob(osp.join(exp_dir, "ckpt-best-*"))[0]
ckpt = torch.load(ckpt_fn, map_location="cpu")
cfg = LazyConfig.load(glob.glob(osp.join(exp_dir, "*.yaml"))[0])
global_set('cfg', cfg)

test_loader = build_dataloader(cfg, split="test")

backbone = instantiate(cfg.model.backbone)
frame_pooler = instantiate(cfg.model.frame_pooler)
head = instantiate(cfg.model.head)
model_cfg = instantiate(cfg.model.model_cfg)
cls = cfg.model._target_
model = cls(backbone=backbone, head=head, frame_pooler=frame_pooler, model_cfg=model_cfg)
model.load_state_dict(ckpt["model"])
model = model.cuda()
model = model.eval()

ret_list = []
evaluater = ValTestEvaluater(cfg)
met, out = evaluate(model, test_loader, evaluater, cfg)
for batch in tqdm(test_loader):
    batch = collection_to_device(batch, "cuda")
    ret_dict = model(**batch, mode="test")
    ret_dict.update(batch)

    gt = ret_dict["gt"]
    boxxes = ret_dict["boxxes"]
    expanded_gt = repeat(gt, "b i -> (b nq) i", nq=300)
    boxxes_flatten = rearrange(boxxes, "b nq i -> (b nq) i")
    ious = calc_iou(boxxes_flatten, expanded_gt)
    ious = rearrange(ious, "(b nq) -> b nq", nq=300)

    attn = torch.load(osp.join(_DEBUG_DIR, "attn.pt"), map_location="cpu")
    ret_dict.update({"attn": attn.detach().cpu().numpy(),
                     "ious": ious.detach().cpu().numpy()})
    ret_list.extend(post_process(ret_dict))

save_pickle(ret_list, osp.join(_DEBUG_DIR, "ret_list.pkl"))
