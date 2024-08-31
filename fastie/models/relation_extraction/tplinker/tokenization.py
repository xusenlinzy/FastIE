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
class DataCollatorForTPLinkerRel:

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
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        num_tag = self.num_labels * 4 + 1
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, num_tag, dtype=torch.long)

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                # SH2OH
                batch_shaking_tag[i, sh, oh, p] = 1
                # OH2SH
                batch_shaking_tag[i, oh, sh, p + self.num_labels] = 1
                # ST2OT
                batch_shaking_tag[i, st, ot, p + self.num_labels * 2] = 1
                # OT2ST
                batch_shaking_tag[i, ot, st, p + self.num_labels * 3] = 1
                # EH2ET
                batch_shaking_tag[i, sh, st, -1] = 1
                batch_shaking_tag[i, oh, ot, -1] = 1

        batch["labels"] = batch_shaking_tag.masked_select(mask[None, :, :, None]).reshape(bs, -1, num_tag)

        return batch


class TPLinkerForRelTokenizer(PreTrainedTokenizerBase):
    def convert_to_features(
        self,
        examples: Mapping,
        label_to_id: dict,
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
        return DataCollatorForTPLinkerRel(
            tokenizer=self,
            ignore_list=ignore_list,
            num_labels=num_labels,
        )


class BertTPLinkerForRelTokenizer(BertTokenizerFast, TPLinkerForRelTokenizer):
    ...


class RoFormerTPLinkerForRelTokenizer(RoFormerTokenizerFast, TPLinkerForRelTokenizer):
    ...
