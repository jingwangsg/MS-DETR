from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from ..common.runtime import paths, flags
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

data = dict(dataset="tacos",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            max_len_video=256,
            target_stride=2,
            datapipe="msat"
)


train = dict(
    prefetch_factor=6,
    max_epochs=100,
    eval_epoch_interval=1,
    batch_size=16,
    optimizer=L(AdamW)(params=None, lr=1e-4),
    val_monitor="val/Rank1@IoU=07",
    clip_grad=2.0
    # lr_scheduler=(StepLR)()
)

from __model.msat import VisualLinguisticTransformer, MultiStageAggregateTransformer, MultiStageHead

model_cfg = dict(d_model=1024,
                 num_clip="${data.video_max_len}",
                 dropout=0.1,
                 loss_cfg=dict(stage_loss=0.3,
                               reg_loss=1.0,
                               iou_loss=200.0,
                               word_mask_loss=0.2,
                               alpha_s=0.25,
                               alpha_m=0.21))

model 