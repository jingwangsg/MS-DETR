from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.runtime import paths, flags
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from ..common.pipelines.clip_pipeline import pipeline as clip_pipeline

data = dict(dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            pipeline_verbose=False,
            **clip_pipeline)

train = dict(
    prefetch_factor=6,
    num_workers=8,
    # prefetch_factor=2,
    # num_workers=0,
    max_epochs=50,
    eval_epoch_interval=1,
    batch_size=16,
    optimizer=L(AdamW)(params=None, lr=1e-4),
    val_monitor="val/Rank1@IoU=07",
    clip_grad=2.0
    # lr_scheduler=(StepLR)()
)

from __model.ms_temporal_detr import MultiScaleTemporalDetr, SegFormerXFPN, SegFormerX, QueryBasedDecoder

d_model = 1024
dropout = 0.1
Lv = 512
Lt = 100
D_TXT = 768
D_VID = 1024

model = L(MultiScaleTemporalDetr)(backbone=L(SegFormerXFPN)(backbone=L(SegFormerX)(
    d_model_in=d_model,
    d_model_lvls=[1024, 1024, 1024, 1024],
    num_head_lvls=[4, 8, 16, 16],
    ff_dim_lvls=[2048, 2048, 2048, 2048],
    sr_ratio_lvls=[4, 4, 2, 1],
    max_vid_len=Lv,
    max_txt_len=Lt,
    input_txt_dim=D_TXT,
    input_vid_dim=D_VID,
    dropout=dropout),
                                                            intermediate_hidden_size=[1024, 1024, 1024, 1024],
                                                            fpn_hidden_size=d_model),
                                  head=L(QueryBasedDecoder)(d_model=d_model,
                                                            nhead=16,
                                                            ff_dim=2048,
                                                            num_query=100,
                                                            num_layers=4,
                                                            num_scales=4,
                                                            pooler_resolution=8,
                                                            dim_init_ref=1,
                                                            dropout=dropout,
                                                            loss_cfg=dict(assign_topk=1,
                                                                          aux_loss=0.01,
                                                                          l1_loss=2.0,
                                                                          focal_loss=1.0)))

# from model.ms_temporal_detr import
