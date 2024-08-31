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
from transformers.tokenization_utils import BatchEncoding


def batchify_rel_labels(batch, features, return_offset_mapping=False):
    """ 关系抽取验证集标签处理 """
    if "text" in features[0].keys():
        batch["texts"] = [feature.pop("text") for feature in features]
    if "target" in features[0].keys():
        batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
    if return_offset_mapping and "offset_mapping" in features[0].keys():
        batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]

    return batch


@dataclass
class DataCollatorForOneRel:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Mapping:
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
            return batchify_rel_labels(batch, features, return_offset_mapping=True)

        bs, seqlen = batch["input_ids"].shape
        seqlens = batch["attention_mask"]
        batch_labels = torch.zeros(bs, self.num_labels, seqlen, seqlen, dtype=torch.long)

        for i, lb in enumerate(labels):
            l = seqlens[i].sum()
            for sh, st, p, oh, ot in lb:
                batch_labels[i, p, sh, oh] = 1
                batch_labels[i, p, sh, ot] = 2
                batch_labels[i, p, st, ot] = 3

            batch_labels[i, :, l:, l:] = -100

        batch["labels"] = batch_labels

        return batch


class OneRelForRelTokenizer(PreTrainedTokenizerBase):
    def convert_to_features(
        self,
        examples: Mapping,
        label_to_id: dict,
        max_length: int = 256,
        text_column_name: str = "text",
        label_column_name: str = "event_list",
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
            for i, spo_list in enumerate(examples[label_column_name]):
                spo = []
                for _sh, _st, p, _oh, _ot in spo_list:
                    try:
                        sh = tokenized_inputs.char_to_token(i, _sh)
                        oh = tokenized_inputs.char_to_token(i, _oh)
                        st = tokenized_inputs.char_to_token(i, _st)
                        ot = tokenized_inputs.char_to_token(i, _ot)
                    except Exception:
                        continue
                    if sh is None or oh is None or st is None or ot is None:
                        continue
                    if isinstance(p, str):
                        p = label_to_id[p]
                    spo.append([sh, st, p, oh, ot])
                labels.append(spo)
            tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def get_collate_fn(self, num_labels: int) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForOneRel(
            tokenizer=self,
            ignore_list=ignore_list,
            num_labels=num_labels,
        )


class BertOneRelForRelTokenizer(BertTokenizerFast, OneRelForRelTokenizer):
    ...


class RoFormerOneRelForRelTokenizer(RoFormerTokenizerFast, OneRelForRelTokenizer):
    ...
