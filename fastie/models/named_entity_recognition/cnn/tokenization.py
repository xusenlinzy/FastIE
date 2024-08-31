from typing import (
    Optional,
    Callable,
    Mapping,
)

import numpy as np
from transformers import (
    BertTokenizerFast,
    PreTrainedTokenizerBase,
    RoFormerTokenizerFast,
)

from .decode_utils import DataCollatorForCnnNer


class CnnNerTokenizer(PreTrainedTokenizerBase):
    def convert_to_features(
        self,
        examples: Mapping,
        label_to_id: dict,
        max_length: int = 256,
        text_column_name: str = "text",
        label_column_name: str = "entities",
        mode: str = "train",
        is_chinese: bool = True,
        with_indices: bool = False,
    ) -> dict:
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "indexes", "label"] if mode == "train" else ["input_ids", "indexes"]
        encoded_inputs = {k: [] for k in input_keys}

        def get_new_ins(bpes, spans, indexes):
            bpes.append(self.sep_token_id)
            cur_word_idx = indexes[-1]
            indexes.append(0)

            if spans is not None:
                matrix = np.zeros((cur_word_idx, cur_word_idx, len(label_to_id)), dtype=np.int8)
                for _ner in spans:
                    s, e, t = _ner
                    if s <= e < cur_word_idx:
                        matrix[s, e, t] = 1
                        matrix[e, s, t] = 1
                return bpes, indexes, matrix

            return bpes, indexes

        for i in range(len(sentences)):
            sentence = sentences[i]
            spans = [] if mode == "train" else None
            _indexes = []
            _bpes = []

            for idx, word in enumerate(sentence):
                __bpes = self.encode(word, add_special_tokens=False)
                _indexes.extend([idx] * len(__bpes))
                _bpes.extend(__bpes)

            indexes = [0] + [i + 1 for i in _indexes]
            bpes = [self.cls_token_id] + _bpes

            if len(bpes) > max_length - 1:
                indexes = indexes[:max_length - 1]
                bpes = bpes[:max_length - 1]

            if mode == "train":
                label = examples[label_column_name][i]
                if with_indices:
                    spans = [
                        (ent["indices"][0], ent["indices"][-1], label_to_id.get(ent["label"]))
                        for ent in label
                    ]
                else:
                    spans = [
                        (ent["start_offset"], ent["end_offset"] - 1, label_to_id.get(ent["label"]))
                        for ent in label
                    ]

            for k, v in zip(input_keys, get_new_ins(bpes, spans, indexes)):
                encoded_inputs[k].append(v)

        return encoded_inputs

    def get_collate_fn(self, num_labels: int) -> Optional[Callable]:
        return DataCollatorForCnnNer(num_labels=num_labels)


class BertCnnNerTokenizer(BertTokenizerFast, CnnNerTokenizer):
    ...


class RoFormerCnnNerTokenizer(RoFormerTokenizerFast, CnnNerTokenizer):
    ...
