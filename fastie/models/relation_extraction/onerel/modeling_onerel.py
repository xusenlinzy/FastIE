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
    Reference:
        🚀 [Code](https://github.com/ssnvxia/OneRel)
    """
)
class OneRelForRelExtraction(PreTrainedModel, RelExtractionDecoder):
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

        self.tag_size = 4
        self.projection_matrix_layer = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 3)
        self.relation_matrix_layer = nn.Linear(self.config.hidden_size * 3, self.config.num_labels * self.tag_size)
        self.entity_pair_dropout = nn.Dropout(getattr(config, "entity_pair_dropout", 0.1))

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
        # encoded_text: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = sequence_output.size()
        # head: [batch_size, seq_len * seq_len, hidden_size]
        head_representation = sequence_output.unsqueeze(2).expand(
            batch_size, seq_len, seq_len, hidden_size
        ).reshape(batch_size, seq_len * seq_len, hidden_size)
        # tail: [batch_size, seq_len * seq_len, hidden_size]
        tail_representation = sequence_output.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, hidden_size * 2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, hidden_size * 3]
        entity_pairs = self.projection_matrix_layer(entity_pairs)
        entity_pairs = self.entity_pair_dropout(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = self.relation_matrix_layer(entity_pairs).reshape(
            batch_size, seq_len, seq_len, self.config.num_labels, self.tag_size
        )

        loss, predictions = None, None
        if labels is not None:
            loss = self.compute_loss([triple_scores, labels])

        if not self.training:
            predictions = self.decode(
                triple_scores, attention_mask, texts, offset_mapping
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
        logits: torch.Tensor,
        masks: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        all_spo_list = []
        batch_size = logits.shape[0]
        masks = tensor_to_numpy(masks)
        logits = tensor_to_numpy(torch.argmax(logits, dim=-1).permute(0, 3, 1, 2))

        id2predicate = self.config.id2label
        for bs in range(batch_size):
            _logits = logits[bs]
            l = masks[bs].sum()
            text, mapping = texts[bs], offset_mapping[bs]

            hs, hts, ts = {}, {}, {}
            for obj, tag in [(hs, 1), (hts, 2), (ts, 3)]:
                for p, h, t in zip(*np.where(_logits == tag)):
                    if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                        continue
                    if p not in obj:
                        obj[p] = []
                    obj[p].append((h, t))

            spoes = set()
            for p in hs.keys() & ts.keys() & hts.keys():
                ht_list = hts[p]
                for sh, oh in hs[p]:
                    for st, ot in ts[p]:
                        if sh <= st and oh <= ot:
                            if (sh, ot) in ht_list:
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
        loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss = loss_fn(logits.permute(0, 4, 3, 1, 2), labels)
        return loss.mean(-1).mean(-1).sum() / logits.size(0)


class BertForOneRelRelExtraction(BertPreTrainedModel, OneRelForRelExtraction):
    ...


class RoFormerForOneRelRelExtraction(RoFormerPreTrainedModel, OneRelForRelExtraction):
    ...
