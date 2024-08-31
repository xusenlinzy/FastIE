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
    PretrainedConfig,
    MODEL_MAPPING,
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
from .modules import (
    HandshakingKernel,
    MultilabelCategoricalCrossentropy,
)


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
    基于`BERT`的`TPLinker`实体识别模型
    + 📖 将`TPLinker`的`shaking`机制引入实体识别模型
    + 📖 对于`token`对采用矩阵上三角展开的方式进行多标签分类

    Reference:
        ⭐️ [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)
        🚀 [Official Code](https://github.com/131250208/TPlinker-joint-extraction)
        🚀 [Simplified Code](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_tplinker_plus.py)
    """
)
class TPLinkerForNer(PreTrainedModel, NerDecoder):
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

        shaking_type = getattr(config, "shaking_type", "cln_plus")
        self.handshaking_kernel = HandshakingKernel(config.hidden_size, shaking_type)
        self.out_dense = nn.Linear(config.hidden_size, config.num_labels)

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

        sequence_output = self.dropout(outputs[0])

        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(sequence_output)
        # shaking_logits: (batch_size, shaking_seq_len, tag_size)
        shaking_logits = self.out_dense(shaking_hiddens)

        loss, predictions = None, None
        if labels is not None:
            loss = self.compute_loss([shaking_logits, labels])

        if not self.training:
            predictions = self.decode(shaking_logits, attention_mask, texts, offset_mapping)

        return SequenceLabelingOutput(
            loss=loss,
            logits=shaking_logits,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        shaking_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        all_entity_list = []
        seq_len = attention_mask.shape[1]
        seqlens, shaking_logits = tensor_to_cpu(attention_mask.sum(1)), tensor_to_cpu(shaking_logits)
        shaking_idx2matrix_idx = [(s, e) for s in range(seq_len) for e in list(range(seq_len))[s:]]
        id2label = self.config.id2label

        for _shaking_logits, l, text, mapping in zip(shaking_logits, seqlens, texts, offset_mapping):
            entities = set()
            l = l.item()
            matrix_spots = self.get_spots_fr_shaking_tag(shaking_idx2matrix_idx, _shaking_logits)

            for e in matrix_spots:
                tag = id2label[e[2]]
                # for an entity, the start position can not be larger than the end pos.
                if e[0] > e[1] or 0 in [e[0], e[1]] or e[0] >= l - 1 or e[1] >= l - 1:
                    continue
                _start, _end = mapping[e[0]][0], mapping[e[1]][1]
                entities.add(
                    (
                        tag,
                        _start,
                        _end,
                        text[_start: _end]
                    )
                )
            all_entity_list.append(entities)

        return all_entity_list

    def get_spots_fr_shaking_tag(
        self,
        shaking_idx2matrix_idx: List[Tuple[int, int]],
        shaking_outputs: torch.Tensor
    ) -> List[Any]:
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start, end, tag), ]
        """
        spots = []
        pred_shaking_tag = (shaking_outputs > self.config.decode_thresh).long()
        nonzero_points = torch.nonzero(pred_shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def compute_loss(self, inputs):
        shaking_logits, labels = inputs[:2]
        loss_fct = MultilabelCategoricalCrossentropy()
        return loss_fct(shaking_logits, labels)


class BertForTPLinkerNer(BertPreTrainedModel, TPLinkerForNer):
    ...


class RoFormerForTPLinkerNer(RoFormerPreTrainedModel, TPLinkerForNer):
    ...
