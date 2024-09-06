import math
import re
from typing import (
    List,
    Union,
    Any,
    Optional,
)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(float(start[1] * end[1]))
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.
    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}

    # 将每一个span的首/尾token的id进行配对（就近匹配，默认没有overlap的情况）
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue

        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue

        if start_id > end_id:
            end_pointer += 1
            continue

    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.
    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def cut_chinese_sent(para):
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


def auto_splitter(input_texts, max_text_len, split_sentence=False):
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


class UIEDecoder(nn.Module):

    keys_to_ignore_on_gpu = ["offset_mapping", "texts"]

    @torch.inference_mode()
    def predict(
        self,
        tokenizer: PreTrainedTokenizer,
        texts: Union[List[str], str],
        schema: Optional[Any] = None,
        batch_size: int = 64,
        max_length: int = 512,
        split_sentence: bool = False,
        position_prob: float = 0.5,
        is_english: bool = False,
        disable_tqdm: bool = True,
    ) -> List[Any]:
        self.eval()
        self.tokenizer = tokenizer
        self.is_english = is_english
        if schema is not None:
            self.set_schema(schema)

        texts = texts
        if isinstance(texts, str):
            texts = [texts]
        return self._multi_stage_predict(
            texts, batch_size, max_length, split_sentence, position_prob, disable_tqdm
        )

    def set_schema(self, schema):
        if isinstance(schema, (dict, str)):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _multi_stage_predict(
        self,
        texts: List[str],
        batch_size: int = 64,
        max_length: int = 512,
        split_sentence: bool = False,
        position_prob: float = 0.5,
        disable_tqdm: bool = True,
    ) -> List[Any]:
        """ Traversal the schema tree and do multi-stage prediction. """
        results = [{} for _ in range(len(texts))]
        if len(texts) < 1 or self._schema_tree is None:
            return results

        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for data in texts:
                    examples.append({"text": data, "prompt": dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, data in zip(node.prefix, texts):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self.is_english:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find("[", 1)].strip()
                                    cls_options = re.search(r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append(
                                {
                                    "text": data,
                                    "prompt": dbc2sbc(prompt)
                                }
                            )
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1

            result_list = self._single_stage_predict(
                examples, batch_size, max_length, split_sentence, position_prob, disable_tqdm
            ) if examples else []
            if not node.parent_relations:
                relations = [[] for _ in range(len(texts))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]["relations"].keys():
                            relations[k][i]["relations"][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(result_list[v[i]])

                new_relations = [[] for _ in range(len(texts))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys() and node.name in relations[i][j]["relations"].keys():
                            for k in range(len(relations[i][j]["relations"][node.name])):
                                new_relations[i].append(relations[i][j]["relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(texts))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self.is_english:
                            prefix[k].append(" of " + result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """ Convert ids to raw text in a single stage. """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start: end], "probability": float(prob[i])}
                else:
                    result = {"text": text[start: end], "start": start, "end": end, "probability": float(prob[i])}

                result_list.append(result)
            results.append(result_list)
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
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

    def _single_stage_predict(
        self,
        inputs: List[dict],
        batch_size: int = 64,
        max_length: int = 512,
        split_sentence: bool = False,
        position_prob: float = 0.5,
        disable_tqdm: bool = True,
    ):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = max_length - len(max(prompts)) - 3

        short_input_texts, input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=split_sentence
        )

        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])
        short_inputs = [
            {
                "text": short_input_texts[i],
                "prompt": short_texts_prompts[i]
            }
            for i in range(len(short_input_texts))
        ]

        encoded_inputs = self.tokenizer(
            text=short_texts_prompts,
            text_pair=short_input_texts,
            stride=2,
            truncation=True,
            max_length=512,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offset_maps = encoded_inputs["offset_mapping"]

        start_prob_concat, end_prob_concat = [], []
        if disable_tqdm:
            batch_iterator = range(0, len(short_input_texts), batch_size)
        else:
            batch_iterator = tqdm(range(0, len(short_input_texts), batch_size), desc="Predicting", unit="batch")
        for batch_start in batch_iterator:
            batch = {
                key:
                    np.array(value[batch_start: batch_start + batch_size], dtype="int64")
                for key, value in encoded_inputs.items() if key not in self.keys_to_ignore_on_gpu
            }

            for k, v in batch.items():
                batch[k] = torch.tensor(v, device=self.device)

            outputs = self(**batch)
            start_prob, end_prob = outputs[0], outputs[1]
            if self.device != torch.device("cpu"):
                start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
            start_prob_concat.append(start_prob.detach().numpy())
            end_prob_concat.append(end_prob.detach().numpy())

        start_prob_concat = np.concatenate(start_prob_concat)
        end_prob_concat = np.concatenate(end_prob_concat)

        start_ids_list = get_bool_ids_greater_than(start_prob_concat, limit=position_prob, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_prob_concat, limit=position_prob, return_prob=True)

        input_ids = encoded_inputs['input_ids'].tolist()
        sentence_ids, probs = [], []
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list, end_ids_list, input_ids, offset_maps):
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map.tolist())
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if not short_result:
                continue
            elif 'start' not in short_result[0].keys() and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            single_results = []
            if is_cls_task:
                cls_options = {}
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] in cls_options:
                        cls_options[short_results[v][0]["text"]][0] += 1
                        cls_options[short_results[v][0]["text"]][1] += short_results[v][0]["probability"]

                    else:
                        cls_options[short_results[v][0]["text"]] = [1, short_results[v][0]["probability"]]

                if cls_options:
                    cls_res, cls_info = max(cls_options.items(), key=lambda x: x[1])
                    concat_results.append(
                        [
                            {"text": cls_res, "probability": cls_info[1] / cls_info[0]}
                        ]
                    )

                else:
                    concat_results.append([])
            else:
                offset = 0
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]["start"] += offset
                            short_results[v][i]["end"] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            f"Invalid schema, value for each key:value pairs should be list or string"
                            f"but {type(v)} received")
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(f"Invalid schema, element should be string or dict, but {type(s)} received")

        return schema_tree


class SchemaTree(object):
    """
    Implementation of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instance of SchemaTree."
        self.children.append(node)
