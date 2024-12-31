import itertools
from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Tuple,
    Set,
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

from .configuration import (
    BertGPLinkerRelConfig,
    RoFormerGPLinkerRelConfig,
)
from .decode_utils import (
    RelExtractionDecoder,
    tensor_to_numpy,
)
from .modules import (
    EfficientGlobalPointer,
    SparseMultilabelCategoricalCrossentropy,
)


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: List[Set[Tuple[str, str, str]]] = None
    groundtruths: List[Set[Tuple[str, str, str]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def get_base_model(config: "PretrainedConfig", **kwargs) -> "PreTrainedModel":
    model_class = MODEL_MAPPING._load_attr_from_module(
        config.model_type, MODEL_MAPPING_NAMES.get(config.model_type)
    )
    return model_class(config, **kwargs)


@add_start_docstrings(
    """
    基于`BERT`的`GPLinker`关系抽取模型
    + 📖 模型的整体思路将三元组抽取分解为实体首尾对应、主体-客体首首对应、主体-客体尾尾对应
    + 📖 通过采用类似多头注意力得分计算的机制将上述三种关系最后映射到一个二维矩阵
    + 📖 每种关系都采用`GlobalPointer`来建模

    Reference:
        ⭐️ [GPLinker：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888)
        🚀 [Code](https://github.com/JunnYu/GPLinker_pytorch)
    """
)
class GPLinkerForRelExtraction(PreTrainedModel, RelExtractionDecoder):
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

        self.hidden_size = config.hidden_size
        # 实体首尾对应，需要相对位置编码且保证首不超过尾
        self.entity_tagger = EfficientGlobalPointer(config.hidden_size, 2, config.head_size)

        # 主体-客体首首对应，不需要相对位置编码和保证首不超过尾
        self.head_tagger = EfficientGlobalPointer(
            config.hidden_size,
            config.num_labels,
            config.head_size,
            use_rope=False,
            tril_mask=False,
        )

        # 主体-客体尾尾对应，不需要相对位置编码和保证首不超过尾
        self.tail_tagger = EfficientGlobalPointer(
            config.hidden_size,
            config.num_labels,
            config.head_size,
            use_rope=False,
            tril_mask=False,
        )

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
        offset_mapping: Optional[List[List[List[int]]]] = None,
        target: Optional[List[Set[Tuple[str, str, str]]]] = None,
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

        # [batch_size, 2, seq_len, seq_len]
        entity_logits = self.entity_tagger(sequence_output, mask=attention_mask)
        # [batch_size, num_predicates, seq_len, seq_len]
        head_logits = self.head_tagger(sequence_output, mask=attention_mask)
        # [batch_size, num_predicates, seq_len, seq_len]
        tail_logits = self.tail_tagger(sequence_output, mask=attention_mask)

        loss, predictions = None, None
        if entity_labels is not None and head_labels is not None and tail_labels is not None:
            entity_loss = self.compute_loss([entity_logits, entity_labels])
            head_loss = self.compute_loss([head_logits, head_labels])
            tail_loss = self.compute_loss([tail_logits, tail_labels])
            loss = (entity_loss + head_loss + tail_loss) / 3

        if not self.training:
            predictions = self.decode(
                entity_logits, head_logits, tail_logits, attention_mask, texts, offset_mapping
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
        entity_logits: torch.Tensor,
        head_logits: torch.Tensor,
        tail_logits: torch.Tensor,
        masks: torch.Tensor,
        texts: List[str],
        offset_mapping: List[List[List[int]]],
    ) -> List[Set[Tuple[str, str, str]]]:
        all_spo_list = []
        batch_size = entity_logits.shape[0]
        masks = tensor_to_numpy(masks)

        entity_logits = tensor_to_numpy(entity_logits)
        head_logits = tensor_to_numpy(head_logits)
        tail_logits = tensor_to_numpy(tail_logits)
        decode_thresh = getattr(self.config, "decode_thresh", 0.0)

        id2predicate = self.config.id2label
        for bs in range(batch_size):
            # 抽取主体和客体
            subjects, objects = set(), set()
            _entity_logits = entity_logits[bs]
            l = masks[bs].sum()
            text, mapping = texts[bs], offset_mapping[bs]
            for r, h, t in zip(*np.where(_entity_logits > decode_thresh)):
                if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                    continue
                if r == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            # 识别对应的关系类型
            spoes = set()
            _head_logits, _tail_logits = head_logits[bs], tail_logits[bs]
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
        preds, target = inputs[:2]
        shape = preds.shape
        target = target[..., 0] * shape[2] + target[..., 1]  # [bsz, heads, num_spoes]
        preds = preds.reshape(shape[0], -1, np.prod(shape[2:]))
        loss_fct = SparseMultilabelCategoricalCrossentropy(mask_zero=True)
        return loss_fct(preds, target.long()).sum(dim=1).mean()


class BertForGPLinkerRelExtraction(BertPreTrainedModel, GPLinkerForRelExtraction):

    config_class = BertGPLinkerRelConfig


class RoFormerForGPLinkerRelExtraction(RoFormerPreTrainedModel, GPLinkerForRelExtraction):

    config_class = RoFormerGPLinkerRelConfig
