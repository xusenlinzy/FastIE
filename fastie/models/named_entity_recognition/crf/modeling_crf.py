from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Any,
    Tuple,
)

import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    RoFormerPreTrainedModel,
    PreTrainedModel,
    PretrainedConfig,
    MODEL_MAPPING,
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
)

from .decode_utils import (
    get_entities,
    sequence_padding,
    tensor_to_cpu,
    NerDecoder,
)
from .modules import CRF


@dataclass
class SequenceLabelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
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
    基于`BERT`的`CRF`实体识别模型
    + 📖 `BERT`编码器提取`token`的语义特征
    + 📖 `CRF`层学习标签之间的约束关系
    """
)
class CrfForNer(PreTrainedModel, NerDecoder):
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

        self.use_lstm = getattr(config, "use_lstm", False)
        mid_hidden_size = getattr(config, "mid_hidden_size", config.hidden_size // 3)
        if self.use_lstm:
            self.mid_layer = nn.LSTM(
                config.hidden_size,
                mid_hidden_size // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=classifier_dropout
            )

        self.classifier = nn.Linear(
            mid_hidden_size if self.use_lstm else config.hidden_size, config.num_labels
        )
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        schemas = sorted(config.schemas)
        bio_labels = ["O"] + [f"B-{l}" for l in schemas] + [f"I-{l}" for l in schemas]
        config.id2label = {int(i): l for i, l in enumerate(bio_labels)}
        return config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        offset_mapping: Optional[List[Any]] = None,
        target: Optional[List[Any]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceLabelingOutput:
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

        # sequence_output = self.dropout(outputs[0])
        sequence_output = outputs[0]
        if self.use_lstm:
            sequence_output, _ = self.mid_layer(sequence_output)

        logits = self.classifier(sequence_output)

        loss, predictions = None, None
        if labels is not None:
            loss = self.compute_loss([logits, labels, attention_mask])

        if not self.training:  # 训练时无需解码
            predictions = self.decode(logits, attention_mask, texts, offset_mapping)

        return SequenceLabelingOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        decode_ids = self.crf.decode(logits, masks.bool()).squeeze(0)  # (batch_size, seq_length)
        decode_ids, masks = tensor_to_cpu(decode_ids), tensor_to_cpu(masks)
        id2label = self.config.id2label

        decode_labels = []
        for text, ids, mask, mapping in zip(texts, decode_ids, masks, offset_mapping):
            decode_label = [id2label[_id.item()] for _id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
            decode_label = get_entities(decode_label)
            decode_label = [
                (
                    l[0],
                    mapping[l[1]][0],
                    mapping[l[2]][1],
                    text[mapping[l[1]][0]: mapping[l[2]][1]]
                )
                for l in decode_label
            ]
            decode_labels.append(set(decode_label))

        return decode_labels

    def compute_loss(self, inputs):

        logits, labels, mask = inputs[:3]
        return -1 * self.crf(emissions=logits, tags=labels, mask=mask.bool())


@add_start_docstrings(
    """
    基于`BERT`的层级`CRF`实体识别模型
    + 📖 `BERT`编码器提取`token`的语义特征
    + 📖 第一阶段`CRF`层学习`BIO`标签之间的约束关系抽取所有实体
    + 📖 第二阶段采用一个线性层对实体进行分类
    """
)
class CascadeCrfForNer(PreTrainedModel, NerDecoder):
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

        self.dense1 = nn.Linear(config.hidden_size, 3)
        self.crf = CRF(num_tags=3, batch_first=True)
        self.dense2 = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        config.id2label = {int(i): l for i, l in enumerate(["O"] + sorted(config.schemas))}
        return config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        offset_mapping: Optional[List[Any]] = None,
        target: Optional[List[Any]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceLabelingOutput:
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
        logits = self.dense1(sequence_output)

        loss, predictions = None, None
        if labels is not None and entity_ids is not None and entity_labels is not None:
            entity_logits = self.get_entity_logits(sequence_output, entity_ids)
            loss = self.compute_loss([logits, entity_logits, entity_labels, labels, attention_mask])

        if not self.training:  # 训练时无需解码
            predictions = self.decode(sequence_output, logits, attention_mask, texts, offset_mapping)

        return SequenceLabelingOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_entity_logits(
        self,
        sequence_output: torch.Tensor,
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        btz, entity_count, _ = entity_ids.shape
        entity_ids = entity_ids.reshape(btz, -1, 1).repeat(1, 1, self.config.hidden_size)
        entity_states = torch.gather(sequence_output, dim=1, index=entity_ids).reshape(
            btz, entity_count, -1, self.config.hidden_size
        )
        entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾 `hidden_states` 的均值
        return self.dense2(entity_states)  # [btz, 实体个数，实体类型数]

    def decode(
        self,
        sequence_output: torch.Tensor,
        logits: torch.Tensor,
        masks: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        decode_ids = self.crf.decode(logits, masks.bool()).squeeze(0)  # (batch_size, seq_length)
        decode_ids, masks = tensor_to_cpu(decode_ids), tensor_to_cpu(masks)
        BIO_MAP = getattr(self.config, "BIO_MAP", {0: "O", 1: "B-ENT", 2: "I-ENT"})
        id2label = self.config.id2label

        entity_ids = []
        for ids, mask in zip(decode_ids, masks):
            decode_label = [BIO_MAP[_id.item()] for _id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
            decode_label = get_entities(decode_label)
            if len(decode_label) > 0:
                entity_ids.append([[l[1], l[2]] for l in decode_label])
            else:
                entity_ids.append([[0, 0]])

        entity_ids = torch.from_numpy(sequence_padding(entity_ids)).to(sequence_output.device)
        entity_logits = self.get_entity_logits(sequence_output, entity_ids)
        entity_preds = torch.argmax(entity_logits, dim=-1)  # [btz, 实体个数]

        decode_labels = []
        entity_ids, entity_preds = tensor_to_cpu(entity_ids), tensor_to_cpu(entity_preds)
        for i, (entities, text, mapping) in enumerate(zip(entity_ids, texts, offset_mapping)):
            tmp = set()
            for j, ent in enumerate(entities):
                s, e, p = ent[0].item(), ent[1].item(), entity_preds[i][j].item()
                if s * e * p != 0:
                    _start, _end = mapping[s][0], mapping[e][1]
                    tmp.add((
                        id2label[p], _start, _end, text[_start: _end]
                    ))
            decode_labels.append(tmp)

        return decode_labels

    def compute_loss(self, inputs):
        logits, entity_logits, entity_labels, labels, mask = inputs[:5]
        loss = -1 * self.crf(emissions=logits, tags=entity_labels, mask=mask.bool())
        loss += 4 * nn.CrossEntropyLoss(ignore_index=0)(
            entity_logits.view(-1, self.config.num_labels), labels.flatten()
        )
        return loss


class BertForCrfNer(BertPreTrainedModel, CrfForNer):
    ...


class BertForCascadeCrfNer(BertPreTrainedModel, CascadeCrfForNer):
    ...


class RoFormerForCrfNer(RoFormerPreTrainedModel, CrfForNer):
    ...


class RoFormerForCascadeCrfNer(RoFormerPreTrainedModel, CascadeCrfForNer):
    ...
