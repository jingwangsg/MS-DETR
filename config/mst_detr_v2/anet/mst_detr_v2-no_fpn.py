from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from config.common.runtime import paths, flags
import os.path as osp
from kn_util.config.common import adamw, reduce_lr_on_plateau
from kn_util.config import eval_str
from models.backbone.segformerx import SegFormerXFPN, SegFormerX
from models.ms_temporal_detr.ms_temporal_detr_v2 import MultiScaleTemporalDetr, QueryBasedDecoder
import torch.nn as nn

data = dict(datapipe="mst_detr_v2",
            dataset="activitynet",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            post_process="post_process_mst_detr",
            max_len_video=512,
            target_stride=4,
            word_mask_rate=0.15,
            vid_hdf5="i3d.hdf5",
            vid_hdf5_key_template="{video_id}")

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="R1@IoU=0.7", is_best="max")

train = dict(num_workers=8,
             num_epochs=30,
             eval_epoch_interval=1,
             batch_size=32,
             optimizer=adamw(lr=1e-4, weight_decay=0.000),
             clip_grad=10.0,
             lr_scheduler=reduce_lr_on_plateau(mode="min", factor=0.8, patience=20),
             val_interval=0.5,
             print_interval=0.2)

model_cfg = dict(d_model=512,
                 ff_dim=512,
                 nhead=16,
                 num_query=30,
                 num_clips=eval_str("${data.max_len_video}//${data.target_stride}"),
                 num_layers_enc=5,
                 num_layers_dec=3,
                 sr_ratio_lvls=[4, 2, 1, 1, 1],
                 use_patch_merge=[True, True, False, False, False],
                 dropout=0.1,
                 iou_cutoff=0.7,
                 w_stage_loss=10.0,
                 w_iou_loss=5.0,
                 w_l1_loss=1.0,
                 w_bd_loss=[0.1, 0.5, 5],
                 w_mask_loss=0.5,
                 topk=10,
                 topk_list=[])

d_model = "${model_cfg.d_model}"
model = L(MultiScaleTemporalDetr)(backbone=None, head=None, frame_pooler=None, model_cfg="${model_cfg}")

model.frame_pooler = L(nn.Identity)()
model.backbone = L(SegFormerX)(d_model_in=d_model,
                               d_model_lvls=eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                               num_head_lvls=eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                               ff_dim_lvls=eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                               input_vid_dim=1024,
                               input_txt_dim=300,
                               max_vid_len="${data.max_len_video}",
                               max_txt_len=100,
                               sr_ratio_lvls="${model_cfg.sr_ratio_lvls}",
                               use_patch_merge="${model_cfg.use_patch_merge}",
                               output_layers=[0, 1, 4])
model.head = L(QueryBasedDecoder)(
    d_model="${model_cfg.d_model}",
    nhead="${model_cfg.nhead}",
    ff_dim="${model_cfg.ff_dim}",
    num_query="${model_cfg.num_query}",
    num_layers="${model_cfg.num_layers_dec}",
    num_scales=3,
    pooler_resolution=32,
)
