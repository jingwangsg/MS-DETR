from models.msat_seg.modeling import TAN, TLocVLBERT, FrameAvgPool
from ...common.runtime import flags, paths
from detectron2.config import LazyCall as L
import os.path as osp
from kn_util.config.common import adamw, reduce_lr_on_plateau
from kn_util.config import eval_str

data = dict(datapipe="msat",
            dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            max_len_video=512,
            target_stride=2,
            vid_hdf5="i3d.hdf5",
            vid_hdf5_key_template="{video_id}",
            word_mask_rate=0.15)

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="R1@IoU=0.7", is_best="max")

train = dict(num_workers=4,
             num_epochs=100,
             batch_size=16,
             optimizer=adamw(lr=1e-4, weight_decay=0.0),
             clip_grad=10.0,
             lr_scheduler=reduce_lr_on_plateau(factor=0.8),
             val_interval=1.0,
             print_interval=0.2)

model_cfg = dict(arch="msat",
                 w_stage_loss=0.3,
                 w_reg_loss=1.0,
                 w_iou_loss=200.0,
                 w_mask_loss=0.25,
                 num_clips=L(eval_str)(s="${data.max_len_video} // ${data.target_stride}"),
                 nms_threshold=0.37,
                 dropout=0.1)

model = L(TAN)(frame_layer=L(FrameAvgPool)(kernel_size=2, stride=2),
               bert_layer=L(TLocVLBERT)(dataset="${data.dataset}",
                                        hidden_size=512,
                                        ff_dim=512,
                                        num_head=32,
                                        num_layers=4,
                                        input_vid_dim=1024,
                                        input_txt_dim=300,
                                        classifier_type="2fc",
                                        classifier_dropout="${model_cfg.dropout}",
                                        classifier_hidden_size=512),
               cfg="${model_cfg}")
