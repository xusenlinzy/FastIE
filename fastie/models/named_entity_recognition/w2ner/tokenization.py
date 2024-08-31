import itertools
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

from .decode_utils import DataCollatorForW2Ner, DIST_TO_IDX


class W2nerTokenizer(PreTrainedTokenizerBase):
    def convert_to_features(
        self,
        examples: Mapping,
        label_to_id: dict,
        max_length: int = 256,
        text_column_name: str = "text",
        label_column_name: str = "entities",
        mode: str = "train",
        is_chinese: bool = True,
    ) -> dict:
        if mode == "train":
            return self.convert_to_train_features(
                examples, label_to_id, max_length,
                text_column_name, label_column_name, is_chinese,
            )
        else:
            return self.convert_to_dev_features(examples, max_length, text_column_name, is_chinese)

    def convert_to_train_features(
        self,
        examples: Mapping,
        label_to_id: dict,
        max_length: int = 256,
        text_column_name: str = "text",
        label_column_name: str = "entities",
        is_chinese: bool = True,
        with_indices: bool = False,
    ) -> dict:
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask", "grid_label"]
        encoded_inputs = {k: [] for k in input_keys}

        for sentence, label in zip(sentences, examples[label_column_name]):
            tokens = [self.tokenize(word) for word in sentence[:max_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _input_ids = self.convert_tokens_to_ids(pieces)
            _input_ids = np.array([self.cls_token_id] + _input_ids + [self.sep_token_id])

            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.int32)
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                start += len(pieces)

            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int32)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i, j in itertools.product(range(length), range(length)):
                _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                    _dist_inputs[i, j]]

            _dist_inputs[_dist_inputs == 0] = 19

            # 标签
            _grid_labels = np.zeros((length, length), dtype=np.int32)
            _grid_mask = np.ones((length, length), dtype=np.int32)

            for entity in label:
                _type = entity["label"]
                if with_indices:
                    indices = entity["indices"]
                else:
                    _start, _end, = entity["start_offset"], entity["end_offset"]
                    indices = list(range(_start, _end))

                if indices[-1] >= max_length - 2:
                    continue

                for i in range(len(indices)):
                    if i + 1 >= len(indices):
                        break
                    _grid_labels[indices[i], indices[i + 1]] = 1
                _grid_labels[indices[-1], indices[0]] = label_to_id[_type] + 2

            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask, _grid_labels]):
                encoded_inputs[k].append(v)

        return encoded_inputs

    def convert_to_dev_features(
        self,
        examples: Mapping,
        max_length: int = 256,
        text_column_name: str = "text",
        is_chinese: bool = True,
    ) -> dict:
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]
        encoded_inputs = {k: [] for k in input_keys}

        for sentence in sentences:
            tokens = [self.tokenize(word) for word in sentence[:max_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]

            _input_ids = self.convert_tokens_to_ids(pieces)
            _input_ids = np.array([self.cls_token_id] + _input_ids + [self.sep_token_id])

            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.int32)
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                start += len(pieces)

            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int32)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i, j in itertools.product(range(length), range(length)):
                _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                    _dist_inputs[i, j]]

            _dist_inputs[_dist_inputs == 0] = 19
            _grid_mask = np.ones((length, length), dtype=np.int32)

            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask]):
                encoded_inputs[k].append(v)

        return encoded_inputs

    def get_collate_fn(self) -> Optional[Callable]:
        return DataCollatorForW2Ner()


class BertW2nerTokenizer(BertTokenizerFast, W2nerTokenizer):
    ...


class RoFormerW2nerTokenizer(RoFormerTokenizerFast, W2nerTokenizer):
    ...
