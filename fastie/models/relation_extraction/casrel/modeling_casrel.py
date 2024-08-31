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
)
from .modules import LayerNorm


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
    基于`BERT`的`CasRel`关系抽取模型
    + 📖 模型的整体结构为级联的跨度标记网络
    + 📖 首先通过一个指针网络抽取出句子中出现的所有主体
    + 📖 然后对于每一个主体和每一个关系类型都通过一个指针网络来抽取出其对应的所有客体

    Reference:
        ⭐️ [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction.](https://aclanthology.org/2020.acl-main.136.pdf)
        🚀 [Code](https://github.com/longlongman/CasRel-pytorch-reimplement)
    """
)
class CasrelForRelExtraction(PreTrainedModel, RelExtractionDecoder):
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

        # 主体标记层
        self.subject_tagger = nn.Linear(config.hidden_size, 2)
        # 条件归一化层（用于融合序列特征和主体特征）
        self.layer_norm = LayerNorm(config.hidden_size, conditional_size=config.hidden_size * 2)
        # 为每一个关系类型设置一个客体标记层
        self.object_tagger = nn.Linear(config.hidden_size, config.num_labels * 2)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        config.id2label = {int(i): l for i, l in enumerate(sorted(config.schemas))}
        return config

    @staticmethod
    def gather(inputs, index):
        """ 取出序列中某个位置的向量 """
        index = index.unsqueeze(-1).unsqueeze(-1)
        index = index.expand(inputs.shape[0], 1, inputs.shape[2])  # [batch_size, 1, hidden_size]
        return inputs.gather(1, index).squeeze(1)  # [batch_size, hidden_size]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        subject_labels: Optional[torch.Tensor] = None,
        object_labels: Optional[torch.Tensor] = None,
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

        loss, predictions = None, None
        if self.training and subject_ids is not None:
            subject_preds = self.subject_tagger(sequence_output)  # [batch_size, seq_len, 2]
            sub_start_output = self.gather(sequence_output, subject_ids[:, 0])  # [batch_size, hidden_size]
            sub_end_output = self.gather(sequence_output, subject_ids[:, 1])
            # 拼接主体开始位置token和结束位置token的embedding
            sub_output = torch.cat([sub_start_output, sub_end_output], 1)  # [batch_size, hidden_size * 2]

            # conditional layernorm
            conditional_output = self.layer_norm((sequence_output, sub_output))
            # [batch_size, seq_len, 2 * num_predicates]
            object_preds = self.object_tagger(conditional_output)
            object_preds = object_preds.reshape(subject_ids.size(0), -1, self.config.num_labels, 2)
            # object_preds = nn.Sigmoid()(object_preds) ** 4 # 放大损失

            if subject_labels is not None and object_labels is not None:
                # subject_preds = nn.Sigmoid()(subject_preds) ** 4
                loss = self.compute_loss(
                    [subject_preds, object_preds, subject_labels, object_labels, attention_mask]
                )

        else:
            predictions = self.decode(texts, sequence_output, attention_mask, offset_mapping)

        return RelationExtractionOutput(
            loss=loss,
            logits=None,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_subjects(self, subject_preds, attention_mask):
        """ 批量数据抽取出每个句子中的所有主体 """
        batch_subjects = []
        for subject_pred, mask in zip(subject_preds, attention_mask):
            length = mask.sum()
            start_thresh = getattr(self.config, "start_thresh", 0.5)
            end_thresh = getattr(self.config, "end_thresh", 0.5)

            start = torch.where(subject_pred[:, 0] > start_thresh)[0]
            end = torch.where(subject_pred[:, 1] > end_thresh)[0]

            subjects = []
            for s in start:
                if s == 0 or s > length - 2:  # 排除[CLS]、[SEP]、[PAD]
                    continue
                e = end[end >= s]
                if len(e) > 0:
                    e = e[0]  # 就近原则
                    if e > length - 2:
                        continue
                    subjects.append((s, e))
            batch_subjects.append(subjects)
        return batch_subjects

    def decode(
        self,
        texts: List[str],
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: List[Any],
    ) -> List[set]:
        """ 解码出批量中每个句子中的三元组 """
        # [batch_size, seq_len, 2]
        subject_preds = torch.sigmoid(self.subject_tagger(sequence_output))
        subject_preds = subject_preds * attention_mask.unsqueeze(-1)  # 排除padding
        batch_subjects = self.extract_subjects(subject_preds, attention_mask)

        decode_labels = []
        for i, subjects in enumerate(batch_subjects):
            spoes = set()
            if subjects:
                subjects = torch.tensor(subjects).to(attention_mask.device)
                # 根据一个句子中出现的所有主体重新构造输入
                masks = attention_mask[i].unsqueeze(0).expand(len(subjects), attention_mask.size(1))
                sequence_outputs = sequence_output[i].unsqueeze(0)
                sequence_outputs = sequence_outputs.expand(
                    len(subjects), sequence_output.size(1), sequence_output.size(2)
                )

                sub_start_output = self.gather(sequence_outputs, subjects[:, 0])
                sub_end_output = self.gather(sequence_outputs, subjects[:, 1])
                sub_output = torch.cat([sub_start_output, sub_end_output], 1)

                conditional_output = self.layer_norm([sequence_outputs, sub_output])
                object_preds = self.object_tagger(conditional_output)
                object_preds = object_preds.reshape(subjects.size(0), -1, self.config.num_labels, 2)

                text, mapping = texts[i], offset_mapping[i]
                spoes = self.extract_spoes(subjects, object_preds, masks, text, mapping)
            decode_labels.append(spoes)
        return decode_labels

    def extract_spoes(
        self, subjects, object_preds, masks, text, mapping
    ) -> set:
        """ 单个句子抽取三元组
        """
        start_thresh = getattr(self.config, "start_thresh", 0.5)
        end_thresh = getattr(self.config, "end_thresh", 0.5)
        id2predicate = self.config.id2label
        spoes = set()
        for subject, object_pred, mask in zip(subjects, object_preds, masks):
            start = torch.where(object_pred[:, :, 0] > start_thresh)
            end = torch.where(object_pred[:, :, 1] > end_thresh)
            for _start, predicate1 in zip(*start):
                _start, predicate1 = _start.item(), predicate1.item()
                if _start == 0 or _start > mask.sum() - 2:
                    continue
                for _end, predicate2 in zip(*end):
                    _end, predicate2 = _end.item(), predicate2.item()
                    if _start <= _end <= mask.sum() - 2 and predicate1 == predicate2:
                        label = id2predicate[predicate1]
                        s_h, s_t = mapping[subject[0].item()][0], mapping[subject[1].item()][1]
                        o_h, o_t = mapping[_start][0], mapping[_end][1]
                        spoes.add(
                            (label, text[s_h: s_t], text[o_h: o_t])
                        )
                        break  # 就近原则
        return spoes

    def compute_loss(self, inputs):
        """ 计算损失函数，总损失为主体损失与客体损失之和 """
        subject_preds, object_preds, subject_labels, object_labels, attention_mask = inputs[:5]
        bce_loss_fct = nn.BCEWithLogitsLoss(reduction="none")

        subject_loss = bce_loss_fct(subject_preds, subject_labels.float()).mean(-1)
        subject_loss = torch.sum(subject_loss * attention_mask.float())
        subject_loss = subject_loss / torch.sum(attention_mask.float())

        object_loss = bce_loss_fct(object_preds, object_labels.float()).mean(-1)
        object_loss = torch.sum(object_loss.sum(2) * attention_mask.float())
        object_loss = object_loss / torch.sum(attention_mask.float())
        return subject_loss + object_loss


class BertForCasrelRelExtraction(BertPreTrainedModel, CasrelForRelExtraction):
    ...


class RoFormerForCasrelRelExtraction(RoFormerPreTrainedModel, CasrelForRelExtraction):
    ...
