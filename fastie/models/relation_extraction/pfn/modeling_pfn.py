import itertools
from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Any,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    BertPreTrainedModel,
    RoFormerPreTrainedModel,
    PretrainedConfig,
    MODEL_MAPPING,
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
)

from .decode_utils import (
    RelExtractionDecoder,
    tensor_to_numpy,
)
from .modules import (
    PfnEncoder,
    NerUnit,
    ReUnit,
)


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_base_model(config: "PretrainedConfig", **kwargs) -> "PreTrainedModel":
    model_class = MODEL_MAPPING._load_attr_from_module(
        config.model_type, MODEL_MAPPING_NAMES.get(config.model_type)
    )
    return model_class(config, **kwargs)


@add_start_docstrings(
    """
    基于`BERT`的`PFN`关系抽取模型
    + 📖 一般的联合抽取模型将实体抽取和关系分类分成两步进行，忽略了两个任务之间的联系
    + 📖 该模型通过分组过滤机制，将隐藏状态分成实体抽取信息、关系抽取信息和共享信息三部分
    + 📖 基于实体抽取信息和共享信息抽取出主语和宾语，基于关系抽取信息和共享信息抽取出对应的关系

    Reference:
        ⭐️ [A Partition Filter Network for Joint Entity and Relation Extraction.](https://aclanthology.org/2021.emnlp-main.17.pdf)
        🚀 [Code](https://github.com/Coopercoppers/PFN)
    """
)
class PfnForRelExtraction(PreTrainedModel, RelExtractionDecoder):
    def __init__(self, config):
        super().__init__(config)
        config = self.apply_config(config)
        self.config = config
        setattr(self, self.base_model_prefix, get_base_model(config))

        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.pfn_hidden_size = getattr(config, "pfn_hidden_size", 300)
        self.feature_extractor = PfnEncoder(self.pfn_hidden_size, config.hidden_size)

        self.ner = NerUnit(self.pfn_hidden_size, 2, classifier_dropout)
        self.re_head = ReUnit(self.pfn_hidden_size, config.num_labels, classifier_dropout)
        self.re_tail = ReUnit(self.pfn_hidden_size, config.num_labels, classifier_dropout)

        self.dropout = nn.Dropout(classifier_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        config.id2label = {int(i): l for i, l in enumerate(sorted(config.schemas))}
        return config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        head_labels: Optional[torch.Tensor] = None,
        tail_labels: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        offset_mapping: Optional[List[Any]] = None,
        target: Optional[List[Any]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> RelationExtractionOutput:
        outputs = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = self.dropout(outputs[0])
        sequence_output = sequence_output.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        mask = attention_mask.transpose(0, 1)  # [seq_len, batch_size]
        h_ner, h_re, h_share = self.feature_extractor(sequence_output)

        ner_score = self.ner(h_ner, h_share, mask).permute(2, 3, 0, 1)
        re_head_score = self.re_head(h_re, h_share, mask).permute(2, 3, 0, 1)
        re_tail_score = self.re_tail(h_share, h_re, mask).permute(2, 3, 0, 1)

        loss, predictions = None, None
        if entity_labels is not None and head_labels is not None and tail_labels is not None:
            entity_loss = self.compute_loss([ner_score, entity_labels])
            head_loss = self.compute_loss([re_head_score, head_labels])
            tail_loss = self.compute_loss([re_tail_score, tail_labels])
            loss = entity_loss + head_loss + tail_loss

        if not self.training:
            predictions = self.decode(
                ner_score, re_head_score, re_tail_score, attention_mask, texts, offset_mapping
            )

        return RelationExtractionOutput(
            loss=loss,
            logits=None,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        ner_score: torch.Tensor,
        re_head_score: torch.Tensor,
        re_tail_score: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        ner_score = tensor_to_numpy(ner_score)
        re_head_score = tensor_to_numpy(re_head_score)
        re_tail_score = tensor_to_numpy(re_tail_score)
        masks = tensor_to_numpy(attention_mask)

        batch_size = len(ner_score)
        decode_thresh = getattr(self.config, "decode_thresh", 0.5)
        id2predicate = self.config.id2label

        all_spo_list = []
        for bs in range(batch_size):
            # 抽取主体和客体
            subjects, objects = set(), set()
            _ner_score, l = ner_score[bs], masks[bs].sum()
            text, mapping = texts[bs], offset_mapping[bs]
            for r, h, t in zip(*np.where(_ner_score > decode_thresh)):
                if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                    continue
                if r == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            # 识别对应的关系类型
            spoes = set()
            _head_logits, _tail_logits = re_head_score[bs], re_tail_score[bs]
            for (sh, st), (oh, ot) in itertools.product(subjects, objects):
                p1s = np.where(_head_logits[:, sh, oh] > decode_thresh)[0]
                p2s = np.where(_tail_logits[:, st, ot] > decode_thresh)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add(
                        (
                            id2predicate[p],
                            text[mapping[sh][0]: mapping[st][1]],
                            text[mapping[oh][0]: mapping[ot][1]]
                        )
                    )
            all_spo_list.append(spoes)
        return all_spo_list

    def compute_loss(self, inputs):
        logits, labels = inputs[:2]
        loss_fct = nn.BCELoss(reduction='sum')
        return loss_fct(logits, labels.float()) / logits.size(-1)


class BertForPfnRelExtraction(BertPreTrainedModel, PfnForRelExtraction):
    ...


class RoFormerForPfnRelExtraction(RoFormerPreTrainedModel, PfnForRelExtraction):
    ...
