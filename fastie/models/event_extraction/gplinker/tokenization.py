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


def batchify_ee_labels(batch, features, return_offset_mapping=False):
    """ 事件抽取验证集标签处理 """
    if "text" in features[0].keys():
        batch["texts"] = [feature.pop("text") for feature in features]
    if "target" in features[0].keys():
        batch["target"] = [[[tuple([a[0], a[1], a[2], int(a[3]), int(a[4])]) for a in e] for e in feature.pop("target")] for feature in features]
    if return_offset_mapping and "offset_mapping" in features[0].keys():
        batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]

    return batch


@dataclass
class DataCollatorForGPLinker:

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
            return batchify_ee_labels(batch, features, return_offset_mapping=True)

        bs = batch["input_ids"].size(0)
        max_head_num = max([len(lb["head_labels"]) for lb in labels])
        max_tail_num = max([len(lb["tail_labels"]) for lb in labels])
        max_argu_num = max([(len(lb) - 1) // 2 for label in labels for lb in label["argu_labels"]])

        batch_argu_labels = torch.zeros(bs, self.num_labels, max_argu_num * 2, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, 1, max_head_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, 1, max_tail_num, 2, dtype=torch.long)

        for b, lb in enumerate(labels):
            # argu_labels
            for argu in lb["argu_labels"]:
                batch_argu_labels[b, argu[0], : len(argu[1:])] = torch.tensor(argu[1:], dtype=torch.long)

            # head_labels
            for ih, (h1, h2) in enumerate(lb["head_labels"]):
                batch_head_labels[b, 0, ih, :] = torch.tensor([h1, h2], dtype=torch.long)

            # tail_labels
            for it, (t1, t2) in enumerate(lb["tail_labels"]):
                batch_tail_labels[b, 0, it, :] = torch.tensor([t1, t2], dtype=torch.long)

        batch["argu_labels"] = batch_argu_labels.reshape(bs, self.num_labels, max_argu_num, 2)
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels

        return batch


class GPLinkerForEventExtractionTokenizer(PreTrainedTokenizerBase):
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
            for b, events in enumerate(examples[label_column_name]):
                argu_labels = {}
                head_labels, tail_labels = set(), set()
                for event in events:
                    for i1, (event_type1, role1, word1, head1, tail1) in enumerate(event):
                        head1, tail1 = int(head1), int(tail1)
                        tp1 = label_to_id["@".join([event_type1, role1])]
                        try:
                            h1 = tokenized_inputs.char_to_token(b, head1)
                            t1 = tokenized_inputs.char_to_token(b, tail1 - 1)
                        except:
                            continue

                        if h1 is None or t1 is None:
                            continue

                        if tp1 not in argu_labels:
                            argu_labels[tp1] = [tp1]
                        argu_labels[tp1].extend([h1, t1])

                        for i2, (event_type2, role2, word2, head2, tail2) in enumerate(event):
                            head2, tail2 = int(head2), int(tail2)
                            if i2 > i1:
                                try:
                                    h2 = tokenized_inputs.char_to_token(b, head2)
                                    t2 = tokenized_inputs.char_to_token(b, tail2 - 1)
                                except:
                                    continue

                                if h2 is None or t2 is None:
                                    continue

                                hl = (min(h1, h2), max(h1, h2))
                                tl = (min(t1, t2), max(t1, t2))

                                if hl not in head_labels:
                                    head_labels.add(hl)

                                if tl not in tail_labels:
                                    tail_labels.add(tl)

                argu_labels = list(argu_labels.values())
                head_labels, tail_labels = list(head_labels), list(tail_labels)

                labels.append(
                    {
                        "argu_labels": argu_labels if len(argu_labels) > 0 else [[0, 0, 0]],
                        "head_labels": head_labels if len(head_labels) > 0 else [[0, 0]],
                        "tail_labels": tail_labels if len(tail_labels) > 0 else [[0, 0]]
                    }
                )

            tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def get_collate_fn(self, num_labels: int) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target", "id"]
        return DataCollatorForGPLinker(
            tokenizer=self,
            ignore_list=ignore_list,
            num_labels=num_labels,
        )


class BertGPLinkerForEventExtractionTokenizer(BertTokenizerFast, GPLinkerForEventExtractionTokenizer):
    ...


class RoFormerGPLinkerForEventExtractionTokenizer(RoFormerTokenizerFast, GPLinkerForEventExtractionTokenizer):
    ...
