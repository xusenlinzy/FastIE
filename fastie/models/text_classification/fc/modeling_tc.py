from typing import (
    Optional,
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
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from .decode_utils import SequenceClassificationDecoder
from .modules import (
    MultiSampleDropout,
    FocalLoss,
    Pooler,
    RDropLoss,
)


def get_base_model(config: "PretrainedConfig", **kwargs) -> "PreTrainedModel":
    model_class = MODEL_MAPPING._load_attr_from_module(
        config.model_type, MODEL_MAPPING_NAMES.get(config.model_type)
    )
    return model_class(config, **kwargs)


class SequenceClassification(PreTrainedModel, SequenceClassificationDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.id2label = {i: l for i, l in enumerate(config.labels)}
        self.pooler_type = getattr(config, "pooler_type", "cls")
        if self.pooler_type not in ["cls_before_pooler", "cls", "first_token_transform"]:
            self.config.output_hidden_states = True

        setattr(self, self.base_model_prefix, get_base_model(config))

        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.pooling = Pooler(self.pooler_type)

        self.use_mdp = getattr(config, "use_mdp", False)
        if self.use_mdp:
            self.classifier = MultiSampleDropout(
                config.hidden_size,
                config.num_labels,
                K=getattr(config, "k", 3),
                p=getattr(config, "p", 0.5),
            )
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
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

        pooled_output = self.dropout(self.pooling(outputs, attention_mask))
        logits = self.classifier(pooled_output)

        loss = self.compute_loss([logits, labels]) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, inputs):
        logits, labels = inputs[:2]
        loss_type = getattr(self.config, "loss_type", "cross_entropy")
        if loss_type == "r-drop":
            alpha = getattr(self.config, "alpha", 4)
            loss_fct = RDropLoss(alpha=alpha, rank="updown")
        elif loss_type == "focal_loss":
            loss_fct = FocalLoss()
        else:
            loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))


class BertForSequenceClassification(BertPreTrainedModel, SequenceClassification):
    ...


class RoFormerForSequenceClassification(RoFormerPreTrainedModel, SequenceClassification):
    ...
