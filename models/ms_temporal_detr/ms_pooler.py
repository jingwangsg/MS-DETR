import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from einops import repeat, rearrange
from typing import List


class RoIAlign1D(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, feat, roi_boxxes_batch: List[torch.Tensor]):
        # B, Lv, _ = feat.shape
        # Lb, _2 = roi_boxxes[i].shape

        for idx, roi_boxxes in enumerate(roi_boxxes_batch):
            zeros_pad = torch.zeros((roi_boxxes.shape[0],),
                                    device=feat.device,
                                    dtype=torch.float)
            ones_pad = zeros_pad + 1
            roi_boxxes_batch[idx] = torch.stack(
                (zeros_pad, roi_boxxes[:, 0], ones_pad, roi_boxxes[:, 1]),
                dim=1)
        feat = repeat(feat, "b lv d -> b d lv i",
                      i=1)  # process it like an image

        align_res = roi_align(feat,
                              roi_boxxes_batch,
                              output_size=(self.output_size, 1),
                              aligned=True).squeeze(-1)
        chunk_sizes = [len(x) for x in roi_boxxes_batch]
        align_res = list(align_res.split(chunk_sizes, dim=0))
        align_res = [x.transpose(1, 2) for x in align_res]

        return align_res


class MultiScaleRoIAlign1D(nn.Module):

    def __init__(self, output_size) -> None:
        super().__init__()
        self.roi_align_1d = RoIAlign1D(output_size=output_size)

    def forward(self, feat_lvls, roi_boxxes_batch):
        # roi_boxxes_batch must be normalized
        align_res_lvls = []
        for feat in feat_lvls:
            Lv = feat.shape[1]
            cur_roi_boxxes_batch = [x * Lv for x in roi_boxxes_batch]
            cur_item = self.roi_align_1d(feat, cur_roi_boxxes_batch)
            align_res_lvls += [cur_item]
        return align_res_lvls


if __name__ == "__main__":
    roi_align_1d = RoIAlign1D(8)
    B = 2
    Lv = 140
    D = 64
    Lb = 4
    feat_lvls = [
        repeat(torch.arange(Lv).float() / Lv, "lv -> b lv d", b=B, d=D).cuda()
        for Lv in [140, 280, 560]
    ]
    # feat = torch.randn(B, Lv, D).cuda()
    boxxes = [(torch.tensor([[0.3, 0.5], [0.2, 0.5], [0.5, 0.7]])).cuda()] * B
    from kn_util.debug import explore_content as EC
    print(EC(feat_lvls, "feat"))
    print(EC(boxxes, "boxxes"))

    # test ms_align_1d
    ms_align_1d = MultiScaleRoIAlign1D(output_size=8)
    align_res_lvls = ms_align_1d(feat_lvls, boxxes)
    print(EC(align_res_lvls, "align_res_lvls"))
    import ipdb; ipdb.set_trace() #FIXME ipdb
