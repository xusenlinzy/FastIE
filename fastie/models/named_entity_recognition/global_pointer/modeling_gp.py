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
    MODEL_MAPPING,
    PretrainedConfig
)
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
)

from .configuration import (
    BertGlobalPointerNerConfig,
    RoFormerGlobalPointerNerConfig,
)
from .decode_utils import (
    tensor_to_cpu,
    NerDecoder,
)
from .modules import (
    GlobalPointer,
    EfficientGlobalPointer,
    MultilabelCategoricalCrossentropy,
    SparseMultilabelCategoricalCrossentropy,
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
    基于`BERT`的`GlobalPointer`实体识别模型
    + 📖 模型的整体思路将实体识别问题转化为每个实体类型下`token`对之间的二分类问题，用统一的方式处理嵌套和非嵌套`NER`
    + 📖 采用多头注意力得分的计算方式来建模`token`对之间的得分
    + 📖 采用旋转式位置编码加入相对位置信息
    + 📖 采用单目标多分类交叉熵推广形式的多标签分类损失函数解决类别不平衡问题
    
    Reference:
    ⭐️ [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
    ⭐️ [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) \\
    🚀 [Code](https://github.com/bojone/GlobalPointer)
    """
)
class GlobalPointerForNer(PreTrainedModel, NerDecoder):
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

        is_efficient = getattr(config, "efficient", True)
        # token对特征的计算方式
        cls = EfficientGlobalPointer if is_efficient else GlobalPointer
        self.global_pointer = cls(
            config.hidden_size,
            config.num_labels,
            config.head_size,
            use_rope=config.use_rope
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
        logits = self.global_pointer(sequence_output, mask=attention_mask)

        loss, predictions = None, None
        if labels is not None:
            sparse = getattr(self.config, 'is_sparse', False)
            loss = self.compute_loss([logits, labels, attention_mask], sparse=sparse)

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
        all_entity_list = []
        seq_lens, logits = tensor_to_cpu(masks.sum(1)), tensor_to_cpu(logits).float()
        id2label = self.config.id2label

        decode_thresh = getattr(self.config, "decode_thresh", 0.0)
        for _logits, l, text, mapping in zip(logits, seq_lens, texts, offset_mapping):
            entity_list = set()
            l = l.item()

            for label_id, start_idx, end_idx in zip(*torch.where(_logits > decode_thresh)):
                label_id, start_idx, end_idx = label_id.item(), start_idx.item(), end_idx.item()
                if start_idx >= (l - 1) or end_idx >= (l - 1) or 0 in [start_idx, end_idx]:
                    continue
                label = id2label[label_id]
                _start, _end = mapping[start_idx][0], mapping[end_idx][1]
                entity_list.add((label, _start, _end, text[_start: _end]))
            all_entity_list.append(set(entity_list))

        return all_entity_list

    def compute_loss(self, inputs, sparse=True):
        """ 便于使用自定义的损失函数 """
        preds, target = inputs[:2]
        shape = preds.shape
        if not sparse:
            loss_fct = MultilabelCategoricalCrossentropy()
            return loss_fct(
                preds.reshape(shape[0] * self.config.num_labels, -1),
                target.reshape(shape[0] * self.config.num_labels, -1)
            )
        else:
            target = target[..., 0] * shape[2] + target[..., 1]  # [bsz, heads, num_spoes]
            preds = preds.reshape(shape[0], -1, np.prod(shape[2:]))
            loss_fct = SparseMultilabelCategoricalCrossentropy(mask_zero=True)
            return loss_fct(preds, target).sum(dim=1).mean()


class BertForGlobalPointerNer(BertPreTrainedModel, GlobalPointerForNer):

    config_class = BertGlobalPointerNerConfig


class RoFormerForGlobalPointerNer(RoFormerPreTrainedModel, GlobalPointerForNer):

    config_class = RoFormerGlobalPointerNerConfig
