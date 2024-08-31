import os
from collections import OrderedDict
from typing import (
    Union,
    TYPE_CHECKING,
)

from transformers import BertTokenizerFast

from .convert import convert_uie_checkpoint
from .modeling_uie import UIEModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


UIE_MODEL_MAPPING = OrderedDict(
    {
        "uie":
            (
                BertTokenizerFast,
                UIEModel,
            ),
    }
)

UIE_TASK_NAMES = list(UIE_MODEL_MAPPING.keys())


def load_uie_tokenizer(
    model_name_or_path: Union[str, os.PathLike],
    task_name: str = "uie",
) -> "PreTrainedTokenizer":
    assert task_name in UIE_TASK_NAMES, f"Avialable task names are {UIE_TASK_NAMES}"
    tokenizer_cls, _ = UIE_MODEL_MAPPING[task_name]
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    return tokenizer


def load_uie_model(
    task_name: str,
    model_name_or_path: Union[str, os.PathLike],
    _fast_init: bool = False,
) -> "PreTrainedModel":
    _, model_cls = UIE_MODEL_MAPPING[task_name]
    model = model_cls.from_pretrained(model_name_or_path, _fast_init=_fast_init)
    model.__class__.register_for_auto_class()
    return model
