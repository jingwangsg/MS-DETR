from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.runtime import paths, flags
import os.path as osp
from kn_util.config.common import adamw, reduce_lr_on_plateau
from kn_util.config import eval_str
from models.ms_temporal_detr import SegFormerXFPN, SegFormerX, QueryBasedDecoder, MultiScaleTemporalDetr

flags["train_no_shuffle"] = True

data = dict(datapipe="mst_detr",
            dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            max_len_video=512,
            target_stride=4,
            word_mask_rate=0.15,
            vid_hdf5="i3d.hdf5",
            vid_hdf5_key_template="{video_id}")

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="R1@IoU=0.7", is_best="max")

train = dict(num_workers=4,
             num_epochs=100,
             eval_epoch_interval=1,
             batch_size=16,
             optimizer=adamw(lr=1e-4, weight_decay=0.000),
             val_monitor="R1@IoU=07",
             clip_grad=10.0,
             lr_scheduler=reduce_lr_on_plateau(factor=0.8, patience=20),
             val_interval=1.0,
             print_interval=0.2)

model_cfg = dict(d_model=512,
                 ff_dim=512,
                 nhead=16,
                 num_layers_enc=4,
                 num_layers_dec=3,
                 sr_ratio_lvls=[4, 4, 2, 1],
                 use_patch_merge=[True, False, True, True],
                 dropout=0.1,
                 w_iou_loss=30.0,
                 w_l1_loss=10.0,
                 w_mask_loss=0.5,
                 w_enc_aux_loss=5.0,
                 w_dec_aux_loss=1.0,
                 sigma_s=0.21,
                 sigma_m=0.25,
                 topk=5)

d_model = "${model_cfg.d_model}"
model = L(MultiScaleTemporalDetr)(backbone=None, head=None, model_cfg="${model_cfg}")
model.backbone = L(SegFormerXFPN)(backbone=None,
                                  output_layer=[0, 2, 3],
                                  intermediate_hidden_size=[d_model]*3,
                                  fpn_hidden_size=d_model)
model.backbone.backbone = L(SegFormerX)(d_model_in=d_model,
                                        d_model_lvls=eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                                        num_head_lvls=eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                                        ff_dim_lvls=eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                                        input_vid_dim=1024,
                                        input_txt_dim=300,
                                        max_vid_len=512,
                                        max_txt_len=100,
                                        sr_ratio_lvls="${model_cfg.sr_ratio_lvls}",
                                        use_patch_merge="${model_cfg.use_patch_merge}")
model.head = L(QueryBasedDecoder)(
    d_model="${model_cfg.d_model}",
    nhead="${model_cfg.nhead}",
    ff_dim="${model_cfg.ff_dim}",
    num_query=100,
    num_layers="${model_cfg.num_layers_dec}",
    num_scales=3,
    pooler_resolution=16,
)
