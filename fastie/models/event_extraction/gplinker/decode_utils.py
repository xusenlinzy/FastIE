import logging
import math
import os
import queue
from dataclasses import dataclass
from multiprocessing import Queue
from typing import (
    List,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
    Mapping,
    Optional,
    Literal,
    Tuple,
)

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
from transformers import is_torch_npu_available
from transformers.utils import ModelOutput

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


os.environ["PYTHONWARNINGS"] = "ignore"
logger = logging.getLogger("FASTIE")


@dataclass
class Argument(ModelOutput):
    role: str = None
    argument: str = None


@dataclass
class Event(ModelOutput):
    event_type: str = None
    trigger: str = None
    arguments: List[Argument] = None


class DedupList(list):
    """ 定义去重的 list """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def isin(event_a, event_b) -> bool:
    """ 判断event_a是否event_b的一个子集 """
    if event_a['event_type'] != event_b["event_type"]:
        return False
    for argu in event_a["arguments"]:
        if argu not in event_b["arguments"]:
            return False
    return True


def neighbors(host, argus, links) -> List:
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


def tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor,)")
    return tensor.detach().cpu()


def tensor_to_numpy(tensor) -> np.ndarray:
    _tensor = tensor_to_cpu(tensor)
    return _tensor.numpy()


def set2json(events: List[List[Tuple[str, str, str, int, int]]]) -> List[Event]:
    event_list = DedupList()
    for event in events:
        final_event = {"event_type": event[0][0], "arguments": DedupList()}
        for argu in event:
            event_type, role = argu[0], argu[1]
            if role != "触发词":
                final_event["arguments"].append({"role": role, "argument": argu[2]})
            else:
                final_event["trigger"] = argu[2]
        event_list = [event for event in event_list if not isin(event, final_event)]
        if not any([isin(final_event, event) for event in event_list]):
            event_list.append(final_event)
    return [
        Event(
            event_type=d.get("event_type"),
            trigger=d.get("trigger"),
            arguments=[Argument(role=a.get("role"), argument=a.get("argument")) for a in d.get("arguments", [])]
        )
        for d in event_list
    ]


class EventExtractionDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: Union[List[str], str],
        batch_size: Optional[int] = 64,
        max_length: Optional[int] = 512,
        language: Optional[str] = "zh",
        show_progress_bar: bool = None,
        device: Optional[str] = None,
    ) -> List[List[Event]]:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        # Cast an individual text to a list with length 1
        if isinstance(texts, str) or not hasattr(texts, "__len__"):
            texts = [texts]

        if language.lower() in ["zh", "zh-cn", "chinese"]:
            infer_inputs = [t.replace(" ", "-") for t in texts]  # 防止空格导致位置预测偏移
        else:
            infer_inputs = texts

        if device is None:
            device = next(self.parameters()).device

        self.to(device)

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Batches", disable=not show_progress_bar):
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
            raise ValueError("The batch received was empty.")
        return inputs

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[Literal["input", "output", "processes"], Any]:
        """启动多进程池，用多个独立进程进行预测
        如果要在多个GPU或CPU上进行预测，建议使用此方法，建议每个GPU只启动一个进程

        Args:
            target_devices (List[str], optional): PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...],
                ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"]. If target_devices is None and CUDA/NPU
                is available, then all available CUDA/NPU devices will be used. If target_devices is None and
                CUDA/NPU is not available, then 4 CPU devices will be used.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                target_devices = ["npu:{}".format(i) for i in range(torch.npu.device_count())]
            else:
                logger.info("CUDA/NPU is not available. Starting 4 CPU workers")
                target_devices = ["cpu"] * 4

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=EventExtractionDecoder._predict_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    @staticmethod
    def stop_multi_process_pool(pool: Dict[Literal["input", "output", "processes"], Any]) -> None:
        """
        Stops all processes started with start_multi_process_pool.

        Args:
            pool (Dict[str, object]): A dictionary containing the input queue, output queue, and process list.

        Returns:
            None
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def predict_multi_process(
        self,
        tokenizer: "PreTrainedTokenizer",
        texts: List[str],
        pool: Dict[Literal["input", "output", "processes"], Any],
        batch_size: int = 64,
        max_length: int = 512,
        language: Optional[str] = "zh",
        chunk_size: Optional[int] = None,
    ) -> List[List[Event]]:
        if chunk_size is None:
            chunk_size = min(math.ceil(len(texts) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(texts) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for text in texts:
            chunk.append(text)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, tokenizer, batch_size, chunk, max_length, language]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, tokenizer, batch_size, chunk, max_length, language])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        return sum([result[1] for result in results_list], [])

    @staticmethod
    def _predict_multi_process_worker(
        target_device: str, model: "EventExtractionDecoder", input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to predict in multi-process setup
        """
        while True:
            try:
                chunk_id, tokenizer, batch_size, chunk, max_length, language = (
                    input_queue.get()
                )
                results = model.predict(
                    tokenizer,
                    chunk,
                    batch_size=batch_size,
                    max_length=max_length,
                    language=language,
                    show_progress_bar=False,
                    device=target_device,
                )

                results_queue.put([chunk_id, results])
            except queue.Empty:
                break
