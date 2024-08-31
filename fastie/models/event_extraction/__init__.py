import os
from collections import OrderedDict
from typing import (
    Union,
    Any,
    List,
    TYPE_CHECKING,
    Callable,
)

from .gplinker import *

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


EVENT_MODEL_MAPPING = OrderedDict(
    {
        "bert-gplinker-event":
            (
                BertGPLinkerForEventExtractionTokenizer,
                BertGPLinkerEventExtractionConfig,
                BertForGPLinkerEventExtraction,
            ),
        "roformer-gplinker-event":
            (
                RoFormerGPLinkerForEventExtractionTokenizer,
                RoFormerGPLinkerEventExtractionConfig,
                RoFormerForGPLinkerEventExtraction,
            ),
    }
)

EVENT_TASK_NAMES = list(EVENT_MODEL_MAPPING.keys())


def load_event_tokenizer(
    model_name_or_path: Union[str, os.PathLike],
    task_name: str = "bert-gplinker-event",
) -> "PreTrainedTokenizer":
    assert task_name in EVENT_TASK_NAMES, f"Avialable task names are {EVENT_TASK_NAMES}"
    tokenizer_cls, _, _ = EVENT_MODEL_MAPPING[task_name]
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    tokenizer.__class__.register_for_auto_class()
    return tokenizer


def get_event_data_collator(
    num_labels: int, tokenizer: "PreTrainedTokenizer",
) -> Callable:
    try:
        data_collator = tokenizer.get_collate_fn(num_labels)
    except TypeError:
        data_collator = tokenizer.get_collate_fn()
    return data_collator


def load_event_model(
    task_name: str,
    model_name_or_path: Union[str, os.PathLike],
    schemas: List[Any],
    _fast_init: bool = False,
    **model_config_kwargs: Any,
) -> "PreTrainedModel":
    kwargs = {"schemas": schemas}
    kwargs.update(model_config_kwargs)

    _, config_cls, model_cls = EVENT_MODEL_MAPPING[task_name]
    config, unused_kwargs = config_cls.from_pretrained(
        model_name_or_path, **kwargs, return_unused_kwargs=True,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)
    config.__class__.register_for_auto_class()

    model = model_cls.from_pretrained(model_name_or_path, config=config, _fast_init=_fast_init)
    model.__class__.register_for_auto_class()

    return model
