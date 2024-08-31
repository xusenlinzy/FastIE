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
from .modules import TransformerDecoderLayer


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
   基于`BERT`的`GRTE`关系抽取模型
    + 📖 模型的整体思路将三元组抽取问题转化为一个表格填充问题，对`token pair`进行多分类
    + 📖 根据实体是否由多个`token`组成将`token pair`之间的关系分成八类
    + 📖 主体-客体-首尾（`S`：单，`M`：多，`H`：首，`T`：尾）：`None`、`SS`、`SMH`、`SMT`、`MSH`、`MST`、`MMH`、`MMT`
    + 📖 全局特征采用`transformer`的带交叉注意力的`encoder`进行迭代学习
    + 📖 采用前向、后向解码的方式进行预测

    Reference:
        ⭐️ [A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.](https://aclanthology.org/2021.emnlp-main.208.pdf)
        🚀 [Official Code](https://github.com/neukg/GRTE)
    """
)
class GrteForRelExtraction(PreTrainedModel, RelExtractionDecoder):
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

        self.Lr_e1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.Lr_e2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.elu = nn.ELU()
        self.Cr = nn.Linear(config.hidden_size, config.num_schemas * config.num_labels)

        self.Lr_e1_rev = nn.Linear(config.num_schemas * config.num_labels, config.hidden_size)
        self.Lr_e2_rev = nn.Linear(config.num_schemas * config.num_labels, config.hidden_size)
        self.e_layer = TransformerDecoderLayer(config)

        # 正交初始化
        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

    @staticmethod
    def apply_config(config):
        config.id2schema = {int(i): v for i, v in enumerate(sorted(config.schemas))}
        config.num_schemas = len(config.schemas)
        tags = ["N/A", "SS", "MSH", "MST", "SMH", "SMT", "MMH", "MMT"]
        config.id2label = {int(i): l for i, l in enumerate(tags)}
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

        bs, l = sequence_output.shape[:2]
        e1 = self.Lr_e1(sequence_output)
        e2 = self.Lr_e2(sequence_output)

        table_logist = None
        for i in range(self.config.rounds):
            h = self.elu(e1.unsqueeze(2).repeat(1, 1, l, 1) * e2.unsqueeze(1).repeat(1, l, 1, 1))
            table_logist = self.Cr(h)
            if i != self.config.rounds - 1:
                table_e1 = table_logist.max(dim=2).values
                table_e2 = table_logist.max(dim=1).values
                e1_ = self.Lr_e1_rev(table_e1)
                e2_ = self.Lr_e2_rev(table_e2)

                e1 = e1 + self.e_layer(e1_, sequence_output, attention_mask)[0]
                e2 = e2 + self.e_layer(e2_, sequence_output, attention_mask)[0]

        logits = table_logist.reshape([bs, l, l, self.config.num_schemas, self.config.num_labels])

        loss, predictions = None, None
        if labels is not None and attention_mask is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            label_masks = attention_mask[:, None, :, None] * attention_mask[:, :, None, None]
            label_masks = label_masks.expand(-1, -1, -1, self.config.num_schemas)
            loss = loss_fct(logits.reshape(-1, self.config.num_labels), labels.reshape([-1]).long())
            loss = (loss * label_masks.reshape([-1])).sum()

        if not self.training:  # 训练时无需解码
            predictions = self.decode(logits, attention_mask, texts, offset_mapping)

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
        logits = tensor_to_cpu(logits.argmax(-1))
        seqlens = tensor_to_cpu(masks.sum(1))
        id2predicate = self.config.id2schema

        triples = []
        for _logits, length, text, mapping in zip(logits, seqlens, texts, offset_mapping):
            tmp = []
            length = length.item()
            for s, e, r in zip(*torch.where(_logits != 0)):
                s, e, r = s.item(), e.item(), r.item()
                if length - 1 <= s or length - 1 <= e or 0 in [s, e]:
                    continue
                tmp.append((s, e, r))

            spoes = set()
            for s, e, r in tmp:
                if _logits[s, e, r] == 1:
                    spoes.add(
                        (
                            id2predicate[r],
                            text[mapping[s][0]: mapping[s][1]],
                            text[mapping[e][0]: mapping[e][1]]
                        )
                    )
                elif _logits[s, e, r] == 4:
                    for s_, e_, r_ in tmp:
                        if r == r_ and _logits[s_, e_, r_] == 5 and s_ == s and e_ > e:
                            spoes.add(
                                (
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s][1]],
                                    text[mapping[e][0]: mapping[e_][1]]
                                )
                            )
                            break
                elif _logits[s, e, r] == 6:
                    for s_, e_, r_ in tmp:
                        if r == r_ and _logits[s_, e_, r_] == 7 and s_ > s and e_ > e:
                            spoes.add(
                                (
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s_][1]],
                                    text[mapping[e][0]: mapping[e_][1]]
                                )
                            )
                            break
                elif _logits[s, e, r] == 2:
                    for s_, e_, r_ in tmp:
                        if r == r_ and _logits[s_, e_, r_] == 3 and s_ > s and e_ == e:
                            spoes.add(
                                (
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s_][1]],
                                    text[mapping[e][0]: mapping[e][1]]
                                )
                            )
                            break
            triples.append(spoes)
        return triples


class BertForGrteRelExtraction(BertPreTrainedModel, GrteForRelExtraction):
    ...


class RoFormerForGrteRelExtraction(RoFormerPreTrainedModel, GrteForRelExtraction):
    ...
