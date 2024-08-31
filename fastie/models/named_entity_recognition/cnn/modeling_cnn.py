from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Any,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from transformers.utils.import_utils import _is_package_available

from .decode_utils import (
    tensor_to_cpu,
    NerDecoder,
    seq_len_to_mask,
    tensor_to_list,
    filter_clashed_by_priority,
)
from .modules import (
    MaskedCNN,
    MultiHeadBiaffine,
)

if _is_package_available("torch_scatter"):
    from torch_scatter import scatter_max


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
class CnnForNer(PreTrainedModel, NerDecoder):
    def __init__(self, config):
        super().__init__(config)
        config = self.apply_config(config)
        setattr(self, self.base_model_prefix, get_base_model(config))

        self.dropout = nn.Dropout(0.4)
        size_embed_dim = getattr(config, "size_embed_dim", 0)
        biaffine_size = getattr(config, "biaffine_size", 200)
        if size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer("span_size_ids", _span_size_ids.long())
            hsz = biaffine_size * 2 + size_embed_dim + 2
        else:
            hsz = biaffine_size * 2 + 2

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(config.hidden_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(config.hidden_size, biaffine_size),
            nn.LeakyReLU(),
        )

        num_heads = getattr(config, "num_heads", 4)
        cnn_hidden_size = getattr(config, "cnn_hidden_size", 200)
        if num_heads > 0:
            self.multi_head_biaffine = MultiHeadBiaffine(
                biaffine_size, cnn_hidden_size, num_heads
            )
        else:
            self.U = nn.Parameter(torch.randn(cnn_hidden_size, biaffine_size, biaffine_size))
            nn.init.xavier_normal_(self.U.data)

        self.W = torch.nn.Parameter(torch.empty(cnn_hidden_size, hsz))
        nn.init.xavier_normal_(self.W.data)

        kernel_size, cnn_depth = getattr(config, "kernel_size", 3), getattr(config, "cnn_depth", 3)
        if cnn_depth > 0:
            self.cnn = MaskedCNN(cnn_hidden_size, cnn_hidden_size, kernel_size=kernel_size, depth=cnn_depth)

        self.fc = nn.Linear(cnn_hidden_size, config.num_labels)

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
        indexes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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

        state = scatter_max(outputs[0], index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)

        if hasattr(self, "U"):
            scores1 = torch.einsum("bxi, oij, byj -> boxy", head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)

        affined_cat = torch.cat(
            [
                self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)
            ],
            dim=-1,
        )

        if hasattr(self, "size_embedding"):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat(
                [affined_cat, self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1
            )

        scores2 = torch.einsum("bmnh, kh -> bkmn", affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1  # bsz x dim x L x L

        if hasattr(self, "cnn"):
            mask = seq_len_to_mask(lengths)
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            u_scores = self.cnn(u_scores, pad_mask)  # bsz, num_labels, max_len, max_len = u_scores.size()
            scores = u_scores + scores

        scores = self.fc(scores.permute(0, 2, 3, 1))

        loss, predictions = None, None
        if labels is not None:
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            mask = labels.ne(-100).float().view(input_ids.size(0), -1)
            loss = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction="none")
            loss = ((loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

        if not self.training:  # 训练时无需解码
            predictions = self.decode(scores, lengths, texts)

        return SequenceLabelingOutput(
            loss=loss,
            logits=scores,
            predictions=predictions,
            groundtruths=target,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def spans_from_upper_triangular(seqlen: int):
        """Spans from the upper triangular area.
        """
        for start in range(seqlen):
            for end in range(start, seqlen):
                yield start, end

    def _decode(
        self,
        scores: torch.Tensor,
        lengths: List[int],
        allow_nested: bool = True,
        thresh: float = 0.5,
    ) -> List[set]:
        batch_chunks = []
        for idx, (_scores, l) in enumerate(zip(scores, lengths)):
            curr_non_mask = scores.new_ones(l, l, dtype=bool).triu()
            tmp_scores = _scores[:l, :l][curr_non_mask]

            confidences, label_ids = tmp_scores, tmp_scores >= thresh
            labels = list(label_ids)
            chunks = [(label, start, end) for label, (start, end) in
                      zip(labels, self.spans_from_upper_triangular(l)) if label != 0]
            confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

            assert len(confidences) == len(chunks)
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
            if len(chunks):
                batch_chunks.append(set([(s, e, l) for l, s, e in chunks]))
            else:
                batch_chunks.append(set())
        return batch_chunks

    def decode(
        self, scores: torch.Tensor, lengths: torch.Tensor, texts: List[str]
    ) -> List[set]:
        all_entity_list = []
        scores, lengths = tensor_to_cpu(torch.sigmoid(scores)), tensor_to_list(lengths)
        scores = (scores + scores.transpose(1, 2)) / 2
        span_pred = scores.max(dim=-1)[0]

        decode_thresh = getattr(self.config, "decode_thresh", 0.5)
        allow_nested = getattr(self.config, "allow_nested", True)
        id2label = self.config.id2label

        span_ents = self._decode(span_pred, lengths, allow_nested=allow_nested, thresh=decode_thresh)
        for span_ent, _scores, text in zip(span_ents, scores, texts):
            entity_set = set()
            for s, e, l in span_ent:
                score = _scores[s, e]
                _type = score.argmax()
                if score[_type] >= decode_thresh:
                    entity_set.add((id2label[_type.item()], s, e + 1, text[s: e + 1]))
            all_entity_list.append(entity_set)

        return all_entity_list


class BertForCnnNer(BertPreTrainedModel, CnnForNer):
    ...


class RoFormerForCnnNer(RoFormerPreTrainedModel, CnnForNer):
    ...
