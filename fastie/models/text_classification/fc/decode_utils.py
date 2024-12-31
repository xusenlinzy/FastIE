import logging
import math
import os
import queue
from multiprocessing import Queue
from typing import (
    List,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
    Mapping,
    Literal,
    Optional,
)

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
from transformers import is_torch_npu_available

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


os.environ["PYTHONWARNINGS"] = "ignore"
logger = logging.getLogger("FASTIE")


class SequenceClassificationDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        text_a: Union[str, List[str]],
        text_b: Union[str, List[str]] = None,
        batch_size: int = 64,
        max_length: int = 512,
        show_progress_bar: bool = None,
        device: Optional[str] = None,
    ) -> List[str]:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if isinstance(text_a, str) or not hasattr(text_a, "__len__"):
            text_a = [text_a]
            if text_b is not None and isinstance(text_b, str):
                text_b = [text_b]

        if device is None:
            device = next(self.parameters()).device

        self.to(device)

        output_list = []
        total_batch = len(text_a) // batch_size + (1 if len(text_a) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Batches", disable=not show_progress_bar):
            batch_text_a = text_a[batch_id * batch_size: (batch_id + 1) * batch_size]
            if text_b is not None:
                batch_text_b = text_b[batch_id * batch_size: (batch_id + 1) * batch_size]
                inputs = tokenizer(
                    batch_text_a,
                    batch_text_b,
                    max_length=max_length,
                    padding=True,
                    truncation="only_second",
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer(
                    batch_text_a,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

            inputs = self._prepare_inputs(inputs)
            outputs = self(**inputs)

            outputs = np.asarray(outputs["logits"].detach().cpu()).argmax(-1)
            output_list.extend(outputs)

        if hasattr(self.config, "id2label"):
            output_list = [self.config.id2label[o] for o in output_list]

        return output_list

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
                target=SequenceClassificationDecoder._predict_multi_process_worker,
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
        chunk_size: Optional[int] = None,
    ) -> List[str]:
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
                    [last_chunk_id, tokenizer, batch_size, chunk, max_length]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, tokenizer, batch_size, chunk, max_length])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        return sum([result[1] for result in results_list], [])

    @staticmethod
    def _predict_multi_process_worker(
        target_device: str, model: "SequenceClassificationDecoder", input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to predict in multi-process setup
        """
        while True:
            try:
                chunk_id, tokenizer, batch_size, chunk, max_length = (
                    input_queue.get()
                )
                results = model.predict(
                    tokenizer,
                    chunk,
                    batch_size=batch_size,
                    max_length=max_length,
                    show_progress_bar=False,
                    device=target_device,
                )

                results_queue.put([chunk_id, results])
            except queue.Empty:
                break
