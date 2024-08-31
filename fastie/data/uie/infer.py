from typing import List

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from ...models.uie.decode_utils import get_bool_ids_greater_than, get_span


def convert_inputs(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    contents: List[str],
    max_length=512,
) -> dict:
    """
    处理输入样本，包括prompt/content的拼接和offset的计算。

    Args:
        tokenizer (tokenizer): tokenizer
        prompts (List[str]): prompt文本列表
        contents (List[str]): content文本列表
        max_length (int): 句子最大长度

    Returns:
        dict -> {
                    'input_ids': tensor([[1, 57, 405, ...]]),
                    'token_type_ids': tensor([[0, 0, 0, ...]]),
                    'attention_mask': tensor([[1, 1, 1, ...]]),
                    'pos_ids': tensor([[0, 1, 2, 3, 4, 5,...]])
                    'offset_mapping': tensor([[[0, 0], [0, 1], [1, 2], [0, 0], [3, 4], ...]])
            }

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/utils.py#L150
    """
    encoded_inputs = tokenizer(
        text=prompts,  # [SEP]前内容
        text_pair=contents,  # [SEP]后内容
        truncation=True,  # 是否截断
        max_length=max_length,  # 句子最大长度
        padding="max_length",  # padding类型
        return_offsets_mapping=True,  # 返回offsets用于计算token_id到原文的映射
    )

    offset_mappings = [[list(x) for x in offset] for offset in encoded_inputs["offset_mapping"]]
    for i in range(len(offset_mappings)):  # offset 重计算
        bias = 0
        for index in range(1, len(offset_mappings[i])):
            mapping = offset_mappings[i][index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = offset_mappings[i][index - 1][1]
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mappings[i][index][0] += bias
            offset_mappings[i][index][1] += bias

    encoded_inputs["offset_mapping"] = offset_mappings

    for k, v in encoded_inputs.items():
        encoded_inputs[k] = torch.tensor(v)

    return encoded_inputs


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


@torch.inference_mode()
def inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
    contents: List[str],
    prompts: List[str],
    max_length=512,
    prob_threshold=0.5
) -> List[str]:
    """
    输入 promot 和 content 列表，返回模型提取结果。

    Args:
        contents (List[str]): 待提取文本列表, e.g. -> [
                                                    '《琅琊榜》是胡歌主演的一部电视剧。',
                                                    '《笑傲江湖》是一部金庸的著名小说。',
                                                    ...
                                                ]
        prompts (List[str]): prompt列表，用于告知模型提取内容, e.g. -> [
                                                                    '主语',
                                                                    '类型',
                                                                    ...
                                                                ]
        max_length (int): 句子最大长度，小于最大长度则padding，大于最大长度则截断。
        prob_threshold (float): sigmoid概率阈值，大于该阈值则二值化为True。

    Returns:
        List: 模型识别结果, e.g. -> [['琅琊榜'], ['电视剧']]
    """
    model.eval()
    inputs = convert_inputs(tokenizer, prompts, contents, max_length=max_length)
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "token_type_ids": inputs["token_type_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }
    outputs = model(**model_inputs)
    output_sp, output_ep = outputs.start_prob, outputs.end_prob
    output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
    start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
    end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

    res = []
    offset_mapping = inputs["offset_mapping"].tolist()
    for start_ids, end_ids, prompt, content, offset_map in zip(
        start_ids_list,
        end_ids_list,
        prompts,
        contents,
        offset_mapping
    ):
        span_set = get_span(start_ids, end_ids)                 # e.g. {(5, 7), (9, 10)}
        current_span_list = []
        for span in span_set:
            if span[0] < len(prompt) + 2:                       # 若答案出现在promot区域，过滤
                continue
            span_text = ""                                      # 答案span
            input_content = prompt + content                    # 对齐token_ids
            for s in range(span[0], span[1] + 1):               # 将 offset map 里 token 对应的文本切回来
                span_text += input_content[offset_map[s][0]: offset_map[s][1]]
            current_span_list.append(span_text)
        res.append(current_span_list)
    return res
