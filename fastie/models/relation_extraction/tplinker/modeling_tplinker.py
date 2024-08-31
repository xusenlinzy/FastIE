import itertools
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
    RelExtractionDecoder,
    tensor_to_cpu,
)
from .modules import (
    HandshakingKernel,
    MultilabelCategoricalCrossentropy,
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
    基于`BERT`的`TPLinker`关系抽取模型
    + 📖 模型的整体思路将三元组抽取问题转化为`token`对之间的链接问题
    + 📖 对于每一个关系类型，主体-客体的链接关系为：首首、尾尾以及实体首尾
    + 📖 对于`token`对采用矩阵上三角展开的方式进行多标签分类

    Reference:
        ⭐️ [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)
        🚀 [Official Code](https://github.com/131250208/TPlinker-joint-extraction)
    """
)
class TPLinkerForRelExtraction(PreTrainedModel, RelExtractionDecoder):
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

        self.handshaking_kernel = HandshakingKernel(config.hidden_size, config.shaking_type)
        self.out_dense = nn.Linear(config.hidden_size, config.num_schemas * 4 + 1)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        config.num_schemas = len(config.schemas)
        link_types = [
            "SH2OH",  # subject head to object head
            "OH2SH",  # object head to subject head
            "ST2OT",  # subject tail to object tail
            "OT2ST",  # object tail to subject tail
        ]
        tags = [
            "=".join([rel, lk])
            for lk, rel in itertools.product(link_types, sorted(config.schemas))
        ]
        tags.append("DEFAULT=EH2ET")
        config.id2label = {idx: t for idx, t in enumerate(tags)}
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

        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(sequence_output)
        # shaking_logits: (batch_size, shaking_seq_len, tag_size)
        shaking_logits = self.out_dense(shaking_hiddens)

        loss, predictions = None, None
        if labels is not None:
            loss = self.compute_loss([shaking_logits, labels])

        if not self.training:
            predictions = self.decode(shaking_logits, attention_mask, texts, offset_mapping)

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
        shaking_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        offset_mapping: List[Any],
    ) -> List[set]:
        all_spo_list = []
        seq_len = attention_mask.shape[1]
        seqlens, shaking_logits = tensor_to_cpu(attention_mask.sum(1)), tensor_to_cpu(shaking_logits)
        shaking_idx2matrix_idx = [(s, e) for s in range(seq_len) for e in list(range(seq_len))[s:]]

        id2label = self.config.id2label
        for _shaking_logits, l, text, mapping in zip(shaking_logits, seqlens, texts, offset_mapping):
            head_ind2entities = {}
            spoes = set()
            l = l.item()

            matrix_spots = self.get_spots_fr_shaking_tag(shaking_idx2matrix_idx, _shaking_logits)
            for sp in matrix_spots:
                tag = id2label[sp[2]]
                ent_type, link_type = tag.split("=")
                # for an entity, the start position can not be larger than the end pos.
                if link_type != "EH2ET" or sp[0] > sp[1] or 0 in [sp[0], sp[1]] or sp[0] >= l - 1 or sp[1] >= l - 1:
                    continue

                entity = {
                    "type": ent_type,
                    "tok_span": [sp[0], sp[1]],
                }
                # take ent_head_pos as the key to entity list
                head_key = sp[0]
                if head_key not in head_ind2entities:
                    head_ind2entities[head_key] = []
                head_ind2entities[head_key].append(entity)

            # tail link
            tail_link_memory_set = set()
            for sp in matrix_spots:
                tag = id2label[sp[2]]
                rel, link_type = tag.split("=")

                if link_type == "ST2OT":
                    tail_link_memory = (rel, sp[0], sp[1])
                    tail_link_memory_set.add(tail_link_memory)
                elif link_type == "OT2ST":
                    tail_link_memory = (rel, sp[1], sp[0])
                    tail_link_memory_set.add(tail_link_memory)

            # head link
            for sp in matrix_spots:
                tag = id2label[sp[2]]
                rel, link_type = tag.split("=")

                if link_type == "SH2OH":
                    subj_head_key, obj_head_key = sp[0], sp[1]
                elif link_type == "OH2SH":
                    subj_head_key, obj_head_key = sp[1], sp[0]
                else:
                    continue

                if (
                    subj_head_key not in head_ind2entities
                    or obj_head_key not in head_ind2entities
                ):
                    # no entity start with subj_head_key and obj_head_key
                    continue

                # all entities start with this subject head
                subj_list = head_ind2entities[subj_head_key]
                # all entities start with this object head
                obj_list = head_ind2entities[obj_head_key]

                for subj, obj in itertools.product(subj_list, obj_list):
                    tail_link_memory = (rel, subj["tok_span"][1], obj["tok_span"][1])

                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    spoes.add(
                        (
                            rel,
                            text[
                            mapping[subj["tok_span"][0]][0]: mapping[
                                subj["tok_span"][1]
                            ][1]
                            ],
                            text[
                            mapping[obj["tok_span"][0]][0]: mapping[
                                obj["tok_span"][1]
                            ][1]
                            ],
                        )
                    )
            all_spo_list.append(set(spoes))
        return all_spo_list

    def get_spots_fr_shaking_tag(
        self,
        shaking_idx2matrix_idx: List[Tuple[int, int]],
        shaking_outputs: torch.Tensor,
    ) -> List[Any]:
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start, end, tag), ]
        """
        spots = []
        decode_thresh = getattr(self.config, "decode_thresh", 0.0)
        pred_shaking_tag = (shaking_outputs > decode_thresh).long()
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


class BertForTPLinkerRelExtraction(BertPreTrainedModel, TPLinkerForRelExtraction):
    ...


class RoFormerForTPLinkerRelExtraction(RoFormerPreTrainedModel, TPLinkerForRelExtraction):
    ...
