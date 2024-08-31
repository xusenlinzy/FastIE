import os
from collections import OrderedDict
from typing import (
    Union,
    Any,
    List,
    TYPE_CHECKING,
    Callable,
)

from transformers import (
    BertTokenizerFast,
    RoFormerTokenizerFast,
    BertConfig,
    RoFormerConfig,
    DataCollatorWithPadding,
)

from .fc import *

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


CLS_MODEL_MAPPING = OrderedDict(
    {
        "bert-fc-cls":
            (
                BertTokenizerFast,
                BertConfig,
                BertForSequenceClassification,
            ),
        "roformer-fc-cls":
            (
                RoFormerTokenizerFast,
                RoFormerConfig,
                RoFormerForSequenceClassification,
            ),
    }
)

CLS_TASK_NAMES = list(CLS_MODEL_MAPPING.keys())


def load_cls_tokenizer(
    model_name_or_path: Union[str, os.PathLike],
    task_name: str = "bert-fc-cls",
) -> "PreTrainedTokenizer":
    assert task_name in CLS_TASK_NAMES, f"Avialable task names are {CLS_TASK_NAMES}"
    tokenizer_cls, _, _ = CLS_MODEL_MAPPING[task_name]
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    return tokenizer


def get_cls_data_collator(
    num_labels: int, tokenizer: "PreTrainedTokenizer",
) -> Callable:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def load_cls_model(
    task_name: str,
    model_name_or_path: Union[str, os.PathLike],
    schemas: List[Any],
    _fast_init: bool = False,
    **model_config_kwargs: Any,
) -> "PreTrainedModel":
    kwargs = {"labels": schemas}
    kwargs.update(model_config_kwargs)

    _, config_cls, model_cls = CLS_MODEL_MAPPING[task_name]
    config, unused_kwargs = config_cls.from_pretrained(
        model_name_or_path, **kwargs, return_unused_kwargs=True,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)

    model = model_cls.from_pretrained(model_name_or_path, config=config, _fast_init=_fast_init)
    model.__class__.register_for_auto_class()

    return model
