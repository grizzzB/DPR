#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List

import torch
import transformers
from torch import Tensor as T
from torch import nn


if transformers.__version__.startswith("4"):
    from transformers import BertConfig, BertModel
    from transformers import AdamW
    from transformers import BertTokenizer
    from transformers import RobertaTokenizer
    from transformers import ElectraConfig, ElectraModel, ElectraTokenizer
else:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizer
    from transformers.tokenization_roberta import RobertaTokenizer

from dpr.utils.data_utils import Tensorizer
from dpr.models.biencoder import BiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    '''
    encoder_model_type: hf_bert
    pretrained_model_cfg: bert-base-uncased
    pretrained_file: null
    projection_dim: 0
    sequence_length: 256
    dropout: 0.1
    fix_ctx_encoder: false
    pretrained: true
    '''
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    # bool. default는 false 
    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_electra_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    '''
    encoder_model_type: koelectra
    pretrained_model_cfg: monologg/koelectra-base-v3-discriminator
    projection_dim: 0
    sequence_length: 256
    dropout: 0.1
    fix_ctx_encoder: false
    pretrained: true
    '''
    question_encoder = KoElectraEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = KoElectraEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    # bool. default는 false 
    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_electra_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


# TODO: unify tensorizer init methods
def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def get_electra_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_electra_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return ElectraTensorizer(tokenizer, sequence_length)


def get_bert_tensorizer_p(
    pretrained_model_cfg: str, sequence_length: int, do_lower_case: bool = True, special_tokens: List[str] = []
):
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    if special_tokens:
        _add_special_tokens(tokenizer, special_tokens)
    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("Tokenizer: %s", type(tokenizer))
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code

    assert special_tokens_num < 500
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)


def get_roberta_tensorizer(pretrained_model_cfg: str, do_lower_case: bool, sequence_length: int):
    tokenizer = get_roberta_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    return RobertaTensorizer(tokenizer, sequence_length)
    

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

#### electra tokenizer를 추가하면 될 듯
def get_electra_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    #pretrained_cfg_name == bert-base-uncased
    #conf/encoder/*.yaml. 여기에 저장되어 있음
    return ElectraTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    #pretrained_cfg_name == bert-base-uncased
    #conf/encoder/*.yaml. 여기에 저장되어 있음
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


class KoElectraEncoder(ElectraModel):
    def __init__(self, config, project_dim: int = 0):
        ElectraModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> ElectraModel:
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        '''
        default config
        {
            "architectures": [
                "ElectraForPreTraining"
            ],
            "attention_probs_dropout_prob": 0.1,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "type_vocab_size": 2,
            "vocab_size": 35000,
            "pad_token_id": 0
        }
        '''
        cfg = ElectraConfig.from_pretrained(cfg_name if cfg_name else "monologg/koelectra-base-v3-discriminator")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    # last_hidden_states, hidden_states는 bert는 그대로 넘겨주지만 pooler_output은 변형해서 넘겨줌. 
    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        #token_type_ids: pretrain 할때 [sep] 앞뒤 문장 구분해주는 id. 보통 fine tuning할때 0만 들어간다고 함.
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        # isinstance(data, data_type).data의 data_type이 맞는지 체크하는 함수인듯. 아래는 클래스 이름으로 체크함.
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            logger.info(f"####CHECK####: {type(out)}")
            sequence_output = out.last_hidden_state
            pooled_output = None 
            hidden_states = out.hidden_states
        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        # overwrite pooled_output.
        if isinstance(representation_token_pos, int):
            # sequence_output'shape : (batch_size, sequence_length, hidden_size)
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            #shape : (config.hidden_size, project_dim) 
            pooled_output = self.encode_proj(pooled_output)

        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        # bert encoder
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        # projection. 0이면 없다고 함. 왜 하는 건지? default는 0임.
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        # HF 기본 함수인 듯. from_pretrained에서는 overwrite된다고 함.
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> BertModel:
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        '''
        Bert config 참고용
        {
            "architectures": [
                "BertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.6.0.dev0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }
        '''
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    # last_hidden_states, hidden_states는 bert는 그대로 넘겨주지만 pooler_output은 변형해서 넘겨줌. 
    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        #token_type_ids: pretrain 할때 [sep] 앞뒤 문장 구분해주는 id. 보통 fine tuning할때 0만 들어간다고 함.
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        # isinstance(data, data_type).data의 data_type이 맞는지 체크하는 함수인듯. 아래는 클래스 이름으로 체크함.
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            # classificatin 할때 최종 output으로 쓰는듯
            # BaseModelOutputWithPoolingAndCrossAttentions일때, 값이 있는데 왜 아래는 값을 받으면서 여기선 None인지??
            pooled_output = None 
            #Hidden-states of the model at the output of each layer plus the optional initial embedding outputs
            # 마지막 하나 레이어가 아니라 각 레이어 값을 다 더한 것이라 함.
            hidden_states = out.hidden_states
        # 4버전 아닌 경우, hidden_states를 쓸 때
        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        # 아닐때.
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        # overwrite pooled_output.
        # representation_token_pos. 정확히 뭔지 모르겟음 DPR에서 추가한 변수임
        if isinstance(representation_token_pos, int):
            # sequence_output'shape : (batch_size, sequence_length, hidden_size)
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            # tensor의 index가 어떻게 먹히는지 모르곘음.
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            #shape : (config.hidden_size, project_dim) 
            pooled_output = self.encode_proj(pooled_output)

        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        # pooler만 써서 이렇게 반환 하는듯.
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


# tokenize->tenserize 하는 class. 전처리 용인 듯함.
class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        # 위에 옵션을 다 true로 하면 아래 코드는 없어도 되는게 아닌지? 이상함
        seq_len = self.max_length
        # padding.
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        # remove overflowed tokens.
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class ElectraTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(ElectraTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)

class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)
