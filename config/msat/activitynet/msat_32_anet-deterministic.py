from models.msat.modeling import TAN, TLocVLBERT, FrameAvgPool
from ...common.runtime import flags, paths
from detectron2.config import LazyCall as L
import os.path as osp
from kn_util.config.common import adamw, reduce_lr_on_plateau
from kn_util.config import eval_str

data = dict(datapipe="msat",
            dataset="activitynet",
            dataset_dir=osp.join("${paths.data_dir}", "${data.dataset}"),
            post_process="post_process_msat",
            max_len_video=256,
            target_stride=8,
            vid_hdf5="i3d.hdf5",
            vid_hdf5_key_template="{video_id}",
            word_mask_rate=0.15)

eval = dict(ms=[1, 5], ns=[0.3, 0.5, 0.7], best_monitor="R1@IoU=0.7", is_best="max")

train = dict(num_workers=4,
             num_epochs=100,
             batch_size=16,
             optimizer=adamw(lr=1e-4, weight_decay=0.0),
             clip_grad=10.0,
             lr_scheduler=reduce_lr_on_plateau(factor=0.8, mode="min", patience=10),
             val_interval=0.5,
             print_interval=0.2)

model_cfg = dict(arch="msat",
                 w_stage_loss=0.4,
                 w_reg_loss=1.0,
                 w_iou_loss=10.0,
                 w_mask_loss=0.1,
                 num_clips=eval_str(s="${data.max_len_video} // ${data.target_stride}"),
                 nms_threshold=0.51,
                 dropout=0.0)

model = L(TAN)(frame_layer=L(FrameAvgPool)(kernel_size=8, stride=8),
               bert_layer=L(TLocVLBERT)(dataset="${data.dataset}",
                                        object_word_embed_mode=2,
                                        input_transform_type=1,
                                        visual_size=1024,
                                        hidden_size=512,
                                        num_hidden_layers=6,
                                        num_attention_heads=16,
                                        intermediate_size=512,
                                        hidden_act="gelu",
                                        hidden_dropout_prob=0.0,
                                        attention_probs_dropout_prob=0.0,
                                        max_position_embeddings=512,
                                        type_vocab_size=2,
                                        vocab_size=10728,
                                        initializer_range=0.02,
                                        visual_scale_text_init=1.0,
                                        visual_scale_object_init=1.0,
                                        visual_ln=False,
                                        word_embedding_frozen=False,
                                        with_pooler=True,
                                        BERT_MODEL_NAME='./model/pretrained_model/bert-base-uncased',
                                        BERT_PRETRAINED='',
                                        BERT_PRETRAINED_EPOCH=0,
                                        CLASSIFIER_TYPE="2fc",
                                        CLASSIFIER_PRETRAINED=True,
                                        CLASSIFIER_DROPOUT=0.0,
                                        CLASSIFIER_HIDDEN_SIZE=512,
                                        NO_GROUNDING=True),
               cfg="${model_cfg}")
