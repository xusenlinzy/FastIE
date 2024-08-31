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
    PreTrainedModel,
    BertPreTrainedModel,
    RoFormerPreTrainedModel,
    MODEL_MAPPING,
    PretrainedConfig
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
)

from .decode_utils import (
    tensor_to_cpu,
    NerDecoder,
)
from .modules import SpanLoss


@dataclass
class SpanOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: Optional[torch.FloatTensor] = None
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
    基于`BERT`的`Span`实体识别模型
    + 📖 对于每个`token`分别进行对应实体类型的起始位置判断
    + 📖 分类数目为实体类型数目+1（非实体）
    """
)
class SpanForNer(PreTrainedModel, NerDecoder):
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

        self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
        self.end_fc = nn.Linear(config.hidden_size, config.num_labels)

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
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        offset_mapping: Optional[List[Any]] = None,
        target: Optional[List[Any]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SpanOutput:
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
        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)

        loss, predictions = None, None
        if start_positions is not None and end_positions is not None:
            loss = self.compute_loss([
                start_logits, end_logits, start_positions, end_positions, attention_mask
            ])

        if not self.training:  # 训练时无需解码
            predictions = self.decode(
                start_logits, end_logits, attention_mask, texts, offset_mapping
            )

        return SpanOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        masks: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        start_labels, end_labels = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)
        start_labels, end_labels = tensor_to_cpu(start_labels), tensor_to_cpu(end_labels)
        id2label = self.config.id2label

        decode_labels = []
        seqlens = tensor_to_cpu(masks.sum(1))
        for _starts, _ends, l, text, mapping in zip(start_labels, end_labels, seqlens, texts, offset_mapping):
            l = l.item()
            decode_label = set()
            for i, s in enumerate(_starts):
                s = s.item()
                if s == 0 or i >= l - 1 or i == 0:
                    continue
                for j, e in enumerate(_ends[i:]):
                    e = e.item()
                    if i + j >= l - 1:
                        continue
                    if s == e:
                        _start, _end = mapping[i][0], mapping[i + j][1]
                        decode_label.add((
                            id2label[s],
                            _start,
                            _end,
                            text[_start: _end])
                        )
                        break
            decode_labels.append(decode_label)
        return decode_labels

    def compute_loss(self, inputs):
        start_logits, end_logits, start_positions, end_positions, masks = inputs[:5]
        loss_fct = SpanLoss()
        return loss_fct(preds=(start_logits, end_logits), target=(start_positions, end_positions), masks=masks)


class BertForSpanNer(BertPreTrainedModel, SpanForNer):
    ...


class RoFormerForSpanNer(RoFormerPreTrainedModel, SpanForNer):
    ...
