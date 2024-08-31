from typing import (
    List,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
    Mapping,
)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class SequenceClassificationDecoder(nn.Module):
    @torch.inference_mode()
    def predict(
        self,
        tokenizer: "PreTrainedTokenizer",
        text_a: Union[str, List[str]],
        text_b: Union[str, List[str]] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.eval()
        if isinstance(text_a, str):
            text_a = [text_a]
            if text_b is not None and isinstance(text_b, str):
                text_b = [text_b]

        output_list = []
        total_batch = len(text_a) // batch_size + (1 if len(text_a) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
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
            raise ValueError(
                "The batch received was empty."
            )
        return inputs
