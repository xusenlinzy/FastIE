import math
import re
from collections import defaultdict
from typing import (
    List,
    Tuple,
    Union,
    Mapping,
    Dict,
    Any,
    Set,
    TYPE_CHECKING,
)

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


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


class NerDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: List[str],
        batch_size: int = 64,
        max_length: int = 512,
        split_sentence: bool = False,
    ) -> List[dict]:
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
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = tokenizer(
                infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size],
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )

            batch_inputs["texts"] = short_input_texts[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = self._prepare_inputs(batch_inputs)
            batch_outputs = self(**batch_inputs)
            outputs.extend(batch_outputs["predictions"])
        return self._auto_joiner(outputs, short_input_texts, input_mapping)

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