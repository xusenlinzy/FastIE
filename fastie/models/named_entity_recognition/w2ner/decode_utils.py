import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Union,
    Mapping,
    Dict,
    Any,
    Set,
    TYPE_CHECKING
)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


DIST_TO_IDX = torch.zeros(1000, dtype=torch.int64)
DIST_TO_IDX[1] = 1
DIST_TO_IDX[2:] = 2
DIST_TO_IDX[4:] = 3
DIST_TO_IDX[8:] = 4
DIST_TO_IDX[16:] = 5
DIST_TO_IDX[32:] = 6
DIST_TO_IDX[64:] = 7
DIST_TO_IDX[128:] = 8
DIST_TO_IDX[256:] = 9


def batchify_ner_labels(batch, features, return_offset_mapping=False):
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
class DataCollatorForW2Ner:
    def __call__(self, features: List[Dict[str, Any]]) -> Mapping:
        labels = ([feature.pop("grid_label") for feature in features] if "grid_label" in features[0].keys() else None)
        input_ids = [feature.pop("input_ids") for feature in features]
        input_ids = torch.from_numpy(sequence_padding(input_ids))

        pieces2word = [feature.pop("pieces2word") for feature in features]
        input_lengths = torch.tensor([len(i) for i in pieces2word], dtype=torch.long)
        max_wordlen = torch.max(input_lengths).item()
        max_pieces_len = max([x.shape[0] for x in input_ids])

        batch_size = input_ids.shape[0]
        sub_mat = torch.zeros(batch_size, max_wordlen, max_pieces_len, dtype=torch.long)
        pieces2word = self.fill(pieces2word, sub_mat)

        dist_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        dist_inputs = [feature.pop("dist_inputs") for feature in features]
        dist_inputs = self.fill(dist_inputs, dist_mat)

        mask_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        grid_mask = [feature.pop("grid_mask") for feature in features]
        grid_mask = self.fill(grid_mask, mask_mat)

        batch = {
            "input_ids": input_ids,
            "dist_inputs": dist_inputs,
            "pieces2word": pieces2word,
            "grid_mask": grid_mask,
            "input_lengths": input_lengths,
        }

        if labels is None:  # for test
            return batchify_ner_labels(batch, features)

        labels_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        labels = self.fill(labels, labels_mat)
        batch["grid_labels"] = labels

        return batch

    @staticmethod
    def fill(data, new_data):
        for i, d in enumerate(data):
            if isinstance(d, np.ndarray):
                new_data[i, :len(d), :len(d[0])] = torch.from_numpy(d).long()
            else:
                new_data[i, :len(d), :len(d[0])] = torch.tensor(d, dtype=torch.long)
        return new_data


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode="post"):
    """ Numpy函数，将序列padding到同一长度 """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def tensor_to_cpu(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor,)")
    return tensor.detach().cpu()


def cut_chinese_sent(para: str):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def auto_splitter(input_texts: List[str], max_text_len: int, split_sentence=False):
    """
    Split the raw texts automatically for model inference.
    Args:
        input_texts (List[str]): input raw texts.
        max_text_len (int): cutting length.
        split_sentence (bool): If True, sentence-level split will be performed.
    return:
        short_input_texts (List[str]): the short input texts for model inference.
        input_mapping (dict): mapping between raw text and short input texts.
    """
    input_mapping = {}
    short_input_texts = []
    cnt_short = 0
    for cnt_org, text in enumerate(input_texts):
        sens = cut_chinese_sent(text) if split_sentence else [text]
        for sen in sens:
            lens = len(sen)
            if lens <= max_text_len:
                short_input_texts.append(sen)
                if cnt_org in input_mapping:
                    input_mapping[cnt_org].append(cnt_short)
                else:
                    input_mapping[cnt_org] = [cnt_short]
                cnt_short += 1
            else:
                temp_text_list = [sen[i: i + max_text_len] for i in range(0, lens, max_text_len)]

                short_input_texts.extend(temp_text_list)
                short_idx = cnt_short
                cnt_short += math.ceil(lens / max_text_len)
                temp_text_id = [short_idx + i for i in range(cnt_short - short_idx)]
                if cnt_org in input_mapping:
                    input_mapping[cnt_org].extend(temp_text_id)
                else:
                    input_mapping[cnt_org] = temp_text_id
    return short_input_texts, input_mapping


def set2json(labels: Set) -> Dict:
    """ 将实体集合根据实体类型转换为字典 """
    res = defaultdict(list)
    for _type, _start, _end, _ent in labels:
        dic = {"start": _start, "end": _end, "text": _ent}
        res[_type].append(dic)
    return res


class W2nerDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: Union[List[str], str],
        batch_size: int = 64,
        max_length: int = 512,
        split_sentence: bool = False,
    ) -> List[Dict[str, Any]]:
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        max_predict_len = max_length - 2
        short_input_texts, input_mapping = auto_splitter(
            texts,
            max_predict_len,
            split_sentence=split_sentence
        )
        infer_inputs = [t.replace(" ", "-") for t in short_input_texts]  # 防止空格导致位置预测偏移

        outputs = []
        collate_fn = DataCollatorForW2Ner()
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = collate_fn(
                [self._process(tokenizer, example, max_length) for example in batch_inputs]
            )
            batch_inputs = self._prepare_inputs(batch_inputs)
            batch_outputs = self(**batch_inputs)
            outputs.extend(batch_outputs["predictions"])
        return self._auto_joiner(outputs, short_input_texts, input_mapping)

    def _process(
        self,
        tokenizer: "PreTrainedTokenizer",
        text: str,
        max_length: int,
    ) -> Dict[str, Any]:
        tokens = [tokenizer.tokenize(word) for word in text[:max_length - 2]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _input_ids = tokenizer.convert_tokens_to_ids(pieces)
        _input_ids = np.array([tokenizer.cls_token_id] + _input_ids + [tokenizer.sep_token_id])

        length = len(tokens)
        _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.int32)
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)

        _dist_inputs = np.zeros((length, length), dtype=np.int32)
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i, j in itertools.product(range(length), range(length)):
            _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                _dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19

        _grid_mask = np.ones((length, length), dtype=np.int32)
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]

        encoded_inputs = {k: v for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask])}
        encoded_inputs["text"] = text

        return encoded_inputs

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return {k: self._prepare_input(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, batch: Any) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(batch)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty."
            )
        return inputs

    def _auto_joiner(
        self,
        short_results: List[Set[Tuple]],
        short_inputs: List[str],
        input_mapping: Dict,
    ) -> List[Dict[str, Any]]:
        concat_results = []
        for k, vs in input_mapping.items():
            single_results = {}
            offset = 0
            for i, v in enumerate(vs):
                if i == 0:
                    single_results = short_results[v]
                else:
                    for res in short_results[v]:
                        tmp = res[0], res[1] + offset, res[2] + offset, res[3]
                        single_results.add(tmp)
                offset += len(short_inputs[v])
            single_results = set2json(single_results) if single_results else {}
            concat_results.append(single_results)

        return concat_results
