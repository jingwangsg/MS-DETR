import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling import BertPredictionHeadTransform
from .visual_linguistic_bert import VisualLinguisticBert
from .segformerx import SegFormerX
from omegaconf import OmegaConf
from kn_util.basic import global_get

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class TLocVLBERT(nn.Module):

    def __init__(self, dataset, **config):

        super(TLocVLBERT, self).__init__()

        config = OmegaConf.create(config)
        self.config = config

        vocab = global_get("glove_vocab", None)
        vocab_size = len(vocab[0]) if vocab else 1000

        if dataset == "ActivityNet".lower():
            iou_mask_map = torch.zeros(33, 33).float()
            for i in range(0, 32, 1):
                iou_mask_map[i, i + 1:min(i + 17, 33)] = 1.
            for i in range(0, 32 - 16, 2):
                iou_mask_map[i, range(18 + i, 33, 2)] = 1.
        elif dataset == "TACoS".lower():
            iou_mask_map = torch.zeros(129, 129).float()
            for i in range(0, 128, 1):
                iou_mask_map[i, 1 + i:min(i + 17, 129)] = 1.
            for i in range(0, 128 - 16, 2):
                iou_mask_map[i, range(18 + i, min(33 + i, 129), 2)] = 1.
            for i in range(0, 128 - 32, 4):
                iou_mask_map[i, range(36 + i, min(65 + i, 129), 4)] = 1.
            for i in range(0, 128 - 64, 8):
                iou_mask_map[i, range(72 + i, 129, 8)] = 1.
        else:
            print('DATASET ERROR')
            exit()

        self.register_buffer('iou_mask_map', iou_mask_map)

        self.vlbert = SegFormerX(d_model_in=config.hidden_size,
                                 d_model_lvls=[config.hidden_size] * config.num_layers,
                                 num_head_lvls=[config.num_head] * config.num_layers,
                                 ff_dim_lvls=[config.ff_dim] * config.num_layers,
                                 sr_ratio_lvls=config.sr_ratio_lvls,
                                 use_patch_merge=config.use_patch_merge,
                                 max_vid_len=1030,
                                 max_txt_len=100,
                                 input_txt_dim=config.input_txt_dim,
                                 input_vid_dim=config.input_vid_dim,
                                 dropout=0.1)

        dim = config.hidden_size
        classifier_dropout = config.classifier_dropout
        classifier_hidden_size = config.classifier_hidden_size
        classifier_dropout = config.classifier_dropout

        if config.classifier_type == "2fc":
            self.final_mlp = torch.nn.Sequential(torch.nn.Dropout(classifier_dropout, inplace=False),
                                                 torch.nn.Linear(dim, classifier_hidden_size),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Dropout(classifier_dropout, inplace=False),
                                                 torch.nn.Linear(classifier_hidden_size, vocab_size))
            self.final_mlp_2 = torch.nn.Sequential(
                torch.nn.Dropout(classifier_dropout, inplace=False),
                torch.nn.Linear(dim, dim * 3),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(classifier_dropout, inplace=False),
            )
            self.final_mlp_3 = torch.nn.Sequential(torch.nn.Linear(dim * 3, classifier_hidden_size),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Dropout(classifier_dropout, inplace=False),
                                                   torch.nn.Linear(classifier_hidden_size, 3))
            self.final_mlp_s = torch.nn.Sequential(torch.nn.Linear(dim, classifier_hidden_size),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Dropout(classifier_dropout, inplace=False),
                                                   torch.nn.Linear(classifier_hidden_size, 1))
            self.final_mlp_e = torch.nn.Sequential(torch.nn.Linear(dim, classifier_hidden_size),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Dropout(classifier_dropout, inplace=False),
                                                   torch.nn.Linear(classifier_hidden_size, 1))
            self.final_mlp_c = torch.nn.Sequential(torch.nn.Linear(dim, classifier_hidden_size),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Dropout(classifier_dropout, inplace=False),
                                                   torch.nn.Linear(classifier_hidden_size, 1))

        # elif config.CLASSIFIER_TYPE == 'mlm':
        #     transform = BertPredictionHeadTransform(config.PARAMS)
        #     linear = nn.Linear(config.hidden_size, config.DATASET.ANSWER_VOCAB_SIZE)
        #     self.final_mlp = nn.Sequential(
        #         transform,
        #         nn.Dropout(config.CLASSIFIER_DROPOUT, inplace=False),
        #         linear
        #     )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        # for m in self.final_mlp.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_2.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.final_mlp_3.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def fix_params(self):
        pass

    def forward(self, text_input_feats, text_mask, word_mask, object_visual_feats):
        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_object = self.vlbert(text_input_feats, text_mask, word_mask,
                                                               object_visual_feats)

        logits_text = self.final_mlp(hidden_states_text)
        hidden_states_object = self.final_mlp_2(hidden_states_object)
        hidden_s, hidden_e, hidden_c = torch.split(hidden_states_object, self.config.hidden_size, dim=-1)

        T = hidden_states_object.size(1)
        s_idx = torch.arange(T, device=hidden_states_object.device)
        e_idx = torch.arange(T, device=hidden_states_object.device)
        c_point = hidden_c[:, (0.5 * (s_idx[:, None] + e_idx[None, :])).long().flatten(), :].view(
            hidden_c.size(0), T, T, hidden_c.size(-1))
        s_c_e_points = torch.cat(
            (hidden_s[:, :, None, :].repeat(1, 1, T, 1), c_point, hidden_e[:, None, :, :].repeat(1, T, 1, 1)), -1)
        logits_iou = self.final_mlp_3(s_c_e_points).permute(0, 3, 1, 2).contiguous()

        logits_visual = torch.cat((self.final_mlp_s(hidden_s), self.final_mlp_e(hidden_e), self.final_mlp_c(hidden_c)),
                                  -1)
        # logits_visual = logits_visual.permute(0,2,1).contiguous()

        return logits_text, logits_visual, logits_iou, self.iou_mask_map.clone().detach()
