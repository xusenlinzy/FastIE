from typing import (
    List,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
    Mapping,
)

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class DedupList(list):
    """ 定义去重的 list """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def isin(event_a, event_b):
    """ 判断event_a是否event_b的一个子集 """
    if event_a['event_type'] != event_b['event_type']:
        return False
    for argu in event_a['arguments']:
        if argu not in event_b['arguments']:
            return False
    return True


def neighbors(host, argus, links):
    """ 构建邻集（host节点与其所有邻居的集合） """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _,  h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def tensor_to_cpu(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor,)")
    return tensor.detach().cpu()


def tensor_to_numpy(tensor):
    _tensor = tensor_to_cpu(tensor)
    return _tensor.numpy()


def set2json(events):
    event_list = DedupList()
    for event in events:
        final_event = {
            "event_type": event[0][0],
            "arguments": DedupList()
        }
        for argu in event:
            event_type, role = argu[0], argu[1]
            if role != "触发词":
                final_event["arguments"].append(
                    {
                        "role": role,
                        "argument": argu[2]
                    }
                )
            else:
                final_event["trigger"] = argu[2]
        event_list = [
            event for event in event_list
            if not isin(event, final_event)
        ]
        if not any([isin(final_event, event) for event in event_list]):
            event_list.append(final_event)
    return event_list


class EventExtractionDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: Union[List[str], str],
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        infer_inputs = [t.replace(" ", "-") for t in texts]  # 防止空格导致位置预测偏移

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

            batch_inputs["texts"] = texts[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = self._prepare_inputs(batch_inputs)
            batch_outputs = self(**batch_inputs)
            outputs.extend(batch_outputs["predictions"])
        return [set2json(o) for o in outputs]

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
