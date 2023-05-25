import torch

from torchmetrics import Metric

from misc import calc_iou
from einops import repeat
from torchmetrics import Metric
from typing import List
import numpy as np


class ScalarMeter(Metric):
    higher_is_better = False
    full_state_update = True

    def __init__(self, mode="avg"):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("maxx", default=torch.tensor(-1e10), dist_reduce_fx="max")
        self.add_state("minx", default=torch.tensor(1e10), dist_reduce_fx="min")
        self.mode = mode

    def update(self, val):
        val = torch.zeros_like(self.sum) + val
        self.sum += val
        self.maxx = torch.maximum(val, self.maxx)
        self.minx = torch.minimum(val, self.minx)
        self.n += 1

    def compute(self):
        mode = self.mode
        if mode == "avg":
            return self.sum / self.n
        if mode == "sum":
            return self.sum
        if mode == "max":
            return self.maxx
        if mode == "min":
            return self.minx


class RankMIoUAboveN(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, m, n) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.add_state("hit", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_sample", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update_single(self, pred_bds, gt, scores):
        _, sorted_index = torch.sort(scores, descending=True)
        pred_bds = pred_bds[sorted_index][:self.m]

        Nc, _2 = pred_bds.shape
        expand_gt = repeat(gt, "i -> nc i", nc=Nc)
        ious = calc_iou(pred_bds, expand_gt)
        is_hit = torch.sum(ious >= self.n)
        self.hit += (is_hit > 0).float()
        self.num_sample += 1

    def update(self, pred_bds, scores, gt):
        B = len(pred_bds)
        pred_bds_batch = pred_bds
        gt_batch = gt
        scores_batch = scores

        if isinstance(pred_bds, List) or len(pred_bds.shape) == 3:
            for i in range(B):
                pred_bds = pred_bds_batch[i]
                gt = gt_batch[i]
                scores = scores_batch[i]
                self.update_single(pred_bds, gt, scores)
        else:
            self.update_single(pred_bds, gt, scores)

    def compute(self):
        return self.hit / self.num_sample * 100


class Evaluater:

    def __init__(self, namespace=None, device="cuda") -> None:
        self.metrics = dict()
        self.namespace = namespace
        self.device = device

    def update_scalar(self, key, val, mode="avg"):
        if key not in self.metrics:
            self.metrics[key] = ScalarMeter().to(self.device)
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.metrics[key].update(val)

    def update_scalar_all(self, outputs, modes=dict()):
        for k in outputs:
            mode = "avg"
            if k in modes:
                mode = modes[k]
            self.update_scalar(k, outputs[k], mode=mode)

    def compute_all(self):
        ret_dict = dict()
        for nm, metric in self.metrics.items():
            if self.namespace:
                nm = self.namespace + "/" + nm
            ret_dict[nm] = metric.compute().item()
            metric.reset()

        return ret_dict


class TrainEvaluater(Evaluater):

    def __init__(self, cfg, device="cuda") -> None:
        super().__init__(device=device)
        self.cfg = cfg

    def update_all(self, losses):
        self.update_scalar_all(losses)


class ValTestEvaluater(Evaluater):

    def __init__(self, cfg, device="cuda"):
        super().__init__(device=device)
        self.cfg = cfg
        ms = cfg.eval.ms
        ns = cfg.eval.ns
        for m in ms:
            for n in ns:
                self.metrics[f"R{m}@IoU={n:.1f}"] = RankMIoUAboveN(m=m, n=n).to(self.device)
        self.metrics["mIoU"] = ScalarMeter()

    def update_all(self, outputs):
        for metric_nm, metric in self.metrics.items():
            boxxes = outputs["boxxes"]
            scores = outputs["scores"]
            gt = outputs["gt"]
            if isinstance(metric, RankMIoUAboveN):
                metric.update(boxxes, scores, gt)
            elif isinstance(metric, ScalarMeter):
                max_indices = scores.argmax(dim=1)
                batch_idxs = torch.arange(scores.shape[0], dtype=torch.long, device=gt.device)
                best_boxxes = boxxes[batch_idxs, max_indices]
                ious = calc_iou(best_boxxes, gt)
                for iou in ious:
                    metric.update(iou)
        

        for k in outputs:
            if k.endswith("loss"):
                self.update_scalar(key=k, val=outputs[k])
