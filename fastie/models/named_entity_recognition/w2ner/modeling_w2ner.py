from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Any,
    Tuple,
)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
    W2nerDecoder,
)
from .modules import (
    DilateConvLayer,
    LayerNorm,
    CoPredictor,
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
    基于`BERT`的`W2Ner`实体识别模型
    + 📖 将实体抽取任务统一起来，可以解决嵌套实体和不连续实体的抽取
    + 📖 将单词对关系进行建模，使用卷积、距离嵌入等抽取表格特征

    Reference:
        ⭐️ [Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)
        🚀 [Official Code](https://github.com/ljynlp/W2NER)
    """
)
class W2ner(PreTrainedModel, W2nerDecoder):
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

        dist_emb_size, type_emb_size = getattr(config, "dist_emb_size", 20), getattr(config, "type_emb_size", 20)
        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        lstm_hidden_size = getattr(config, "lstm_hidden_size", 512)
        self.lstm = nn.LSTM(
            config.hidden_size,
            lstm_hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        conv_input_size = lstm_hidden_size + dist_emb_size + type_emb_size
        conv_hidden_size = getattr(config, "conv_hidden_size", 96)
        self.conv = DilateConvLayer(
            conv_input_size,
            conv_hidden_size,
            dilation=[1, 2, 3],
            dropout=0.5,
        )

        biaffine_size = getattr(config, "biaffine_size", 512)
        ffn_hidden_size = getattr(config, "ffn_hidden_size", 288)
        self.predictor = CoPredictor(
            config.num_labels,
            lstm_hidden_size,
            biaffine_size,
            conv_hidden_size * 3,
            ffn_hidden_size,
            0.33,
        )
        self.cln = LayerNorm(lstm_hidden_size, conditional_size=lstm_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def apply_config(config):
        config.id2label = {int(i): l for i, l in enumerate(["NONE", "NNW"] + sorted(config.schemas))}
        return config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pieces2word: Optional[torch.Tensor] = None,
        dist_inputs: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        grid_mask: Optional[torch.Tensor] = None,
        grid_labels: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
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

        if self.config.use_last_4_layers:
            sequence_output = torch.stack(outputs[2][-4:], dim=-1).mean(-1)
        else:
            sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)  # [batch_size, seq_len, hidden_size]

        length = pieces2word.size(1)
        min_value = torch.min(sequence_output).item()

        # Max pooling word representations from pieces
        sequence_output = sequence_output.unsqueeze(1).expand(-1, length, -1, -1)
        sequence_output = torch.masked_fill(sequence_output, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(sequence_output, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=input_lengths.max())

        cln = self.cln([word_reps.unsqueeze(2), word_reps])

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask.clone().long())
        reg_inputs = tril_mask + grid_mask.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.conv(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask.eq(0).unsqueeze(-1), 0.0)

        logits = self.predictor(word_reps, word_reps, conv_outputs)

        loss, predictions = None, None
        if grid_labels is not None:
            loss = self.compute_loss([logits, grid_labels, grid_mask])

        if not self.training:
            predictions = self.decode(logits, input_lengths, texts)

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
        input_lengths: torch.Tensor,
        texts: List[str],
    ) -> List[set]:
        decode_labels = []
        logits, input_lengths = tensor_to_cpu(logits.argmax(-1)), tensor_to_cpu(input_lengths)
        id2label = self.config.id2label

        for _logits, length, text in zip(logits, input_lengths, texts):
            forward_dict, head_dict, ht_type_dict = {}, {}, {}
            for i in range(length):
                for j in range(i + 1, length):
                    if _logits[i, j].item() == 1:  # NNW
                        if i not in forward_dict:
                            forward_dict[i] = [j]
                        else:
                            forward_dict[i].append(j)

            for i in range(length):
                for j in range(i, length):
                    if _logits[j, i].item() > 1:  # THW
                        ht_type_dict[(i, j)] = _logits[j, i].item()
                        if i not in head_dict:
                            head_dict[i] = {j}
                        else:
                            head_dict[i].add(j)

            predicts = []

            def find_entity(key, entity, tails):
                entity.append(key)
                if key in tails:
                    predicts.append(entity.copy())
                if key not in forward_dict:
                    entity.pop()
                    return
                for k in forward_dict[key]:
                    find_entity(k, entity, tails)
                entity.pop()

            for head in head_dict:
                find_entity(head, [], head_dict[head])

            entities = set()
            for _entity in predicts:
                entities.add(
                    (
                        id2label[ht_type_dict[(_entity[0], _entity[-1])]],
                        _entity[0],
                        _entity[-1] + 1,
                        "".join([text[i] for i in _entity]),
                    )
                )
            decode_labels.append(entities)

        return decode_labels

    def compute_loss(self, inputs):
        logits, labels, mask = inputs[:3]
        active_loss = mask.view(-1) == 1
        active_logits = logits.reshape(-1, self.config.num_labels)

        loss_fct = nn.CrossEntropyLoss()
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        return loss_fct(active_logits, active_labels.long())


class BertForW2ner(BertPreTrainedModel, W2ner):
    ...


class RoFormerForW2ner(RoFormerPreTrainedModel, W2ner):
    ...
