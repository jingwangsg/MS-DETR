from __future__ import division
import torch
import torch.nn as nn
from .modeling import BertLayerNorm, BertEncoder, BertPooler, ACT2FN, BertOnlyMLMHead
import numpy as np
import math

# todo: add this to config
# NUM_SPECIAL_WORDS = 1000

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=116):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class VisualLinguisticBert(BaseModel):
    def __init__(self, dataset, config, language_pretrained_model_path=None):
        super(VisualLinguisticBert, self).__init__(config)

        self.config = config

        # embeddings
        self.mask_embeddings = nn.Embedding(1, config.hidden_size) 
        self.word_mapping = nn.Linear(300, config.hidden_size)    # 300 is the dim of glove vector
        self.text_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.text_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.visual_embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        if dataset == "ActivityNet".lower():
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=116)
        elif dataset == "TACoS".lower():
            self.postion_encoding = PositionalEncoding(config.hidden_size, n_position=194)
        else:
            print('DATASET ERROR')
            exit()

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.visual_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.visual_size, config.hidden_size)
        if config.visual_ln:
            self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.encoder = BertEncoder(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            print('load language pretrained model')
            self.load_language_pretrained_model(language_pretrained_model_path)

    def forward(self,
                text_input_feats,
                text_mask,
                word_mask,
                object_visual_embeddings,
                output_all_encoded_layers=False,
                output_attention_probs=False):

        # get seamless concatenate embeddings and mask
        
        # sync
        # from kn_util.debug import sync_diff
        # debug_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment-v3/_DEBUG"
        # sync_diff(text_mask, "text_mask", debug_dir)
        # sync_diff(text_input_feats, "text_input_feats", debug_dir)
        # sync_diff(word_mask, "word_mask", debug_dir)
        # sync_diff(object_visual_embeddings, "object_visual_embeddings", debug_dir)

        text_embeddings, visual_embeddings = self.embedding(text_input_feats,
                                                            text_mask, word_mask,
                                                            object_visual_embeddings)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = text_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        # extended_attention_mask = 1.0 - extended_attention_mask
        # extended_attention_mask[extended_attention_mask != 0] = float('-inf')

        if output_attention_probs:
            encoded_layers, attention_probs = self.encoder(text_embeddings,
                                                            visual_embeddings,
                                                            extended_attention_mask,
                                                            output_all_encoded_layers=output_all_encoded_layers,
                                                            output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.encoder(text_embeddings,
                                           visual_embeddings,
                                           extended_attention_mask,
                                           output_all_encoded_layers=output_all_encoded_layers,
                                           output_attention_probs=output_attention_probs)

        # from kn_util.debug import sync_diff
        # debug_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment-v3/_DEBUG"
        # sync_diff(encoded_layers, "encoded_layers", debug_dir)
            
        # sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        if output_all_encoded_layers:
            encoded_layers_text = []
            encoded_layers_object = []
            for encoded_layer in encoded_layers:
                encoded_layers_text.append(encoded_layer[0])
                encoded_layers_object.append(encoded_layer[1])
            if output_attention_probs:
                attention_probs_text = []
                attention_probs_object = []
                for attention_prob in attention_probs:
                    attention_probs_text.append(attention_prob[0])
                    attention_probs_object.append(attention_prob[1])
                return encoded_layers_text, encoded_layers_object, attention_probs_text, attention_probs_object
            else:
                return encoded_layers_text, encoded_layers_object
        else:
            encoded_layers = encoded_layers[-1]
            if output_attention_probs:
                attention_probs = attention_probs[-1]
                return encoded_layers[0], encoded_layers[1], attention_probs[0], attention_probs[1]
            else:
                return encoded_layers[0], encoded_layers[1]

    def embedding(self,
                  text_input_feats,
                  text_mask,
                  word_mask,
                  object_visual_embeddings):

        text_linguistic_embedding = self.word_mapping(text_input_feats)
        text_input_feats_temp = text_input_feats.clone()
        mask_word_mean = text_mask
        if self.training:
            text_input_feats_temp[word_mask>0] = 0
            mask_word_mean = text_mask * (1. - word_mask)
            _zero_id = torch.zeros(text_linguistic_embedding.shape[:2], dtype=torch.long, device=text_linguistic_embedding.device)
            text_linguistic_embedding[word_mask>0] = self.mask_embeddings(_zero_id)[word_mask>0]

        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        if self.config.visual_ln:
            object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)

        embeddings = torch.cat([object_visual_embeddings, text_linguistic_embedding], dim=1)
        embeddings = self.postion_encoding(embeddings)
        visual_embeddings, text_embeddings = torch.split(embeddings, [object_visual_embeddings.size(1),text_linguistic_embedding.size(1)], 1)

        text_embeddings = self.text_embedding_LayerNorm(text_embeddings)
        text_embeddings = self.text_embedding_dropout(text_embeddings)

        visual_embeddings = self.visual_embedding_LayerNorm(visual_embeddings)
        visual_embeddings = self.visual_embedding_dropout(visual_embeddings)

        return text_embeddings, visual_embeddings

    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        unexpected_keys = []
        for k, v in pretrained_state_dict.items():
            if k.startswith('bert.'):
                k = k[len('bert.'):]
            elif k.startswith('roberta.'):
                k = k[len('roberta.'):]
            else:
                unexpected_keys.append(k)
                continue
            if 'gamma' in k:
                k = k.replace('gamma', 'weight')
            if 'beta' in k:
                k = k.replace('beta', 'bias')
            if k.startswith('encoder.'):
                k_ = k[len('encoder.'):]
                if k_ in self.encoder.state_dict():
                    encoder_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            elif k.startswith('embeddings.'):
                k_ = k[len('embeddings.'):]
                if k_ == 'word_embeddings.weight':
                    self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                            device=self.word_embeddings.weight.data.device)
                elif k_ == 'position_embeddings.weight':
                    self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                device=self.position_embeddings.weight.data.device)
                elif k_ == 'token_type_embeddings.weight':
                    self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                        dtype=self.token_type_embeddings.weight.data.dtype,
                        device=self.token_type_embeddings.weight.data.device)
                    if v.size(0) == 1:
                        # Todo: roberta token type embedding
                        self.token_type_embeddings.weight.data[1] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        self.token_type_embeddings.weight.data[2] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)

                elif k_.startswith('LayerNorm.'):
                    k__ = k_[len('LayerNorm.'):]
                    if k__ in self.embedding_LayerNorm.state_dict():
                        embedding_ln_pretrained_state_dict[k__] = v
                    else:
                        unexpected_keys.append(k)
                else:
                    unexpected_keys.append(k)
            elif self.config.with_pooler and k.startswith('pooler.'):
                k_ = k[len('pooler.'):]
                if k_ in self.pooler.state_dict():
                    pooler_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            else:
                unexpected_keys.append(k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)


class VisualLinguisticBertForPretraining(VisualLinguisticBert):
    def __init__(self, config, language_pretrained_model_path=None,
                 with_rel_head=True, with_mlm_head=True, with_mvrc_head=True):

        super(VisualLinguisticBertForPretraining, self).__init__(config, language_pretrained_model_path=None)

        self.with_rel_head = with_rel_head
        self.with_mlm_head = with_mlm_head
        self.with_mvrc_head = with_mvrc_head
        if with_rel_head:
            self.relationsip_head = VisualLinguisticBertRelationshipPredictionHead(config)
        if with_mlm_head:
            self.mlm_head = BertOnlyMLMHead(config, self.word_embeddings.weight)
        if with_mvrc_head:
            self.mvrc_head = VisualLinguisticBertMVRCHead(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.pos_embedding_frozen:
            for p in self.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_visual_embeddings,
                text_mask,
                object_vl_embeddings,
                object_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False):

        text_out, object_out, pooled_rep = super(VisualLinguisticBertForPretraining, self).forward(
            text_input_ids,
            text_token_type_ids,
            text_visual_embeddings,
            text_mask,
            object_vl_embeddings,
            object_mask,
            output_all_encoded_layers=False,
            output_text_and_object_separately=True
        )

        if self.with_rel_head:
            relationship_logits = self.relationsip_head(pooled_rep)
        else:
            relationship_logits = None
        if self.with_mlm_head:
            mlm_logits = self.mlm_head(text_out)
        else:
            mlm_logits = None
        if self.with_mvrc_head:
            mvrc_logits = self.mvrc_head(object_out)
        else:
            mvrc_logits = None

        return relationship_logits, mlm_logits, mvrc_logits

    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        relationship_head_pretrained_state_dict = {}
        mlm_head_pretrained_state_dict = {}
        unexpected_keys = []
        for _k, v in pretrained_state_dict.items():
            if _k.startswith('bert.') or _k.startswith('roberta.'):
                k = _k[len('bert.'):] if _k.startswith('bert.') else _k[len('roberta.'):]
                if 'gamma' in k:
                    k = k.replace('gamma', 'weight')
                if 'beta' in k:
                    k = k.replace('beta', 'bias')
                if k.startswith('encoder.'):
                    k_ = k[len('encoder.'):]
                    if k_ in self.encoder.state_dict():
                        encoder_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
                elif k.startswith('embeddings.'):
                    k_ = k[len('embeddings.'):]
                    if k_ == 'word_embeddings.weight':
                        self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                                device=self.word_embeddings.weight.data.device)
                    elif k_ == 'position_embeddings.weight':
                        self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                    device=self.position_embeddings.weight.data.device)
                    elif k_ == 'token_type_embeddings.weight':
                        self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        if v.size(0) == 1:
                            # Todo: roberta token type embedding
                            self.token_type_embeddings.weight.data[1] = v[0].to(
                                dtype=self.token_type_embeddings.weight.data.dtype,
                                device=self.token_type_embeddings.weight.data.device)
                    elif k_.startswith('LayerNorm.'):
                        k__ = k_[len('LayerNorm.'):]
                        if k__ in self.embedding_LayerNorm.state_dict():
                            embedding_ln_pretrained_state_dict[k__] = v
                        else:
                            unexpected_keys.append(_k)
                    else:
                        unexpected_keys.append(_k)
                elif self.config.with_pooler and k.startswith('pooler.'):
                    k_ = k[len('pooler.'):]
                    if k_ in self.pooler.state_dict():
                        pooler_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
            elif _k.startswith('cls.seq_relationship.') and self.with_rel_head:
                k_ = _k[len('cls.seq_relationship.'):]
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.relationsip_head.caption_image_relationship.state_dict():
                    relationship_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            elif (_k.startswith('cls.predictions.') or _k.startswith('lm_head.')) and self.with_mlm_head:
                k_ = _k[len('cls.predictions.'):] if _k.startswith('cls.predictions.') else _k[len('lm_head.'):]
                if _k.startswith('lm_head.'):
                    if 'dense' in k_ or 'layer_norm' in k_:
                        k_ = 'transform.' + k_
                    if 'layer_norm' in k_:
                        k_ = k_.replace('layer_norm', 'LayerNorm')
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.mlm_head.predictions.state_dict():
                    mlm_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            else:
                unexpected_keys.append(_k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)
        if self.with_rel_head and len(relationship_head_pretrained_state_dict) > 0:
            self.relationsip_head.caption_image_relationship.load_state_dict(relationship_head_pretrained_state_dict)
        if self.with_mlm_head:
            self.mlm_head.predictions.load_state_dict(mlm_head_pretrained_state_dict)


class VisualLinguisticBertMVRCHeadTransform(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHeadTransform, self).__init__(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

        self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


class VisualLinguisticBertMVRCHead(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHead, self).__init__(config)

        self.transform = VisualLinguisticBertMVRCHeadTransform(config)
        self.region_cls_pred = nn.Linear(config.hidden_size, config.visual_region_classes)
        self.apply(self.init_weights)

    def forward(self, hidden_states):

        hidden_states = self.transform(hidden_states)
        logits = self.region_cls_pred(hidden_states)

        return logits


class VisualLinguisticBertRelationshipPredictionHead(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertRelationshipPredictionHead, self).__init__(config)

        self.caption_image_relationship = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, pooled_rep):

        relationship_logits = self.caption_image_relationship(pooled_rep)

        return relationship_logits

