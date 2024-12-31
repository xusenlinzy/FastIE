from dataclasses import dataclass
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any,
    Callable,
    Mapping,
)

import torch
from transformers import (
    BertTokenizerFast,
    PreTrainedTokenizerBase,
    RoFormerTokenizerFast,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding

from .decode_utils import sequence_padding


def batchify_ner_labels(
    batch: BatchEncoding, features: List[Dict[str, Any]], return_offset_mapping: bool = False
) -> BatchEncoding:
    """ 命名实体识别验证集标签处理 """
    if "text" in features[0].keys():
        batch["texts"] = [feature.pop("text") for feature in features]
    if "target" in features[0].keys():
        batch["target"] = [
            {tuple([t[0], int(t[1]), int(t[2]), t[3]])
             for t in feature.pop("target")} for feature in features
        ]
    if return_offset_mapping and "offset_mapping" in features[0].keys():
        batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
    return batch


@dataclass
class DataCollatorForGlobalPointerNer:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    is_sparse: Optional[bool] = True
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in self.ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            return batchify_ner_labels(batch, features, return_offset_mapping=True)

        bs, seqlen = batch["input_ids"].shape
        if self.is_sparse:
            batch_labels = []
            for lb in labels:
                label = [set() for _ in range(self.num_labels)]
                for start, end, tag in lb:
                    label[tag].add((start, end))
                for l in label:
                    if not l:  # 至少要有一个标签
                        l.add((0, 0))  # 如果没有则用0填充
                label = sequence_padding([list(l) for l in label])
                batch_labels.append(label)
            batch_labels = torch.from_numpy(sequence_padding(batch_labels, seq_dims=2))
        else:
            batch_labels = torch.zeros(bs, self.num_labels, seqlen, seqlen, dtype=torch.long)
            for i, lb in enumerate(labels):
                for start, end, tag in lb:
                    batch_labels[i, tag, start, end] = 1  # 0为"O"

        batch["labels"] = batch_labels

        return batch


class GlobalPointerForNerTokenizer(PreTrainedTokenizerBase):
    def convert_to_features(
        self,
        examples: Mapping,
        label_to_id: Dict[str, int],
        max_length: int = 256,
        text_column_name: str = "text",
        label_column_name: str = "entities",
        mode: str = "train",
        is_chinese: bool = True,
    ) -> BatchEncoding:
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        tokenized_inputs = self(
            sentences,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )

        if mode == "train":
            labels = []
            for i, entity_list in enumerate(examples[label_column_name]):
                res = []
                for _ent in entity_list:
                    try:
                        start = tokenized_inputs.char_to_token(i, _ent["start_offset"])
                        end = tokenized_inputs.char_to_token(i, _ent["end_offset"] - 1)
                    except Exception:
                        continue
                    if start is None or end is None:
                        continue
                    res.append([start, end, label_to_id[_ent["label"]]])
                labels.append(res)
            tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def get_collate_fn(self, num_labels: int) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForGlobalPointerNer(
            tokenizer=self,
            ignore_list=ignore_list,
            num_labels=num_labels,
        )


class BertGlobalPointerForNerTokenizer(BertTokenizerFast, GlobalPointerForNerTokenizer):
    ...


class RoFormerGlobalPointerForNerTokenizer(RoFormerTokenizerFast, GlobalPointerForNerTokenizer):
    ...
