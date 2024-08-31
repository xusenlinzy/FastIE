import os
from collections import OrderedDict
from typing import (
    Union,
    Any,
    List,
    TYPE_CHECKING,
    Callable,
)

from .casrel import *
from .gplinker import *
from .grte import *
from .onerel import *
from .pfn import *
from .tplinker import *

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


REL_MODEL_MAPPING = OrderedDict(
    {
        "bert-gplinker-relation":
            (
                BertGPLinkerForRelTokenizer,
                BertGPLinkerRelConfig,
                BertForGPLinkerRelExtraction,
            ),
        "roformer-gplinker-relation":
            (
                RoFormerGPLinkerForRelTokenizer,
                RoFormerGPLinkerRelConfig,
                RoFormerForGPLinkerRelExtraction,
            ),
        "bert-grte-relation":
            (
                BertGrteForRelTokenizer,
                BertGrteRelConfig,
                BertForGrteRelExtraction,
            ),
        "roformer-grte-relation":
            (
                RoFormerGrteForRelTokenizer,
                RoFormerGrteRelConfig,
                RoFormerForGrteRelExtraction,
            ),
        "bert-tplinker-relation":
            (
                BertTPLinkerForRelTokenizer,
                BertTPLinkerRelConfig,
                BertForTPLinkerRelExtraction,
            ),
        "roformer-tplinker-relation":
            (
                RoFormerTPLinkerForRelTokenizer,
                RoFormerTPLinkerRelConfig,
                RoFormerForTPLinkerRelExtraction,
            ),
        "bert-casrel-relation":
            (
                BertCasrelForRelTokenizer,
                BertCasrelRelConfig,
                BertForCasrelRelExtraction,
            ),
        "roformer-casrel-relation":
            (
                RoFormerCasrelForRelTokenizer,
                RoFormerCasrelRelConfig,
                RoFormerForCasrelRelExtraction,
            ),
        "bert-pfn-relation":
            (
                BertPfnForRelTokenizer,
                BertPfnRelConfig,
                BertForPfnRelExtraction,
            ),
        "roformer-pfn-relation":
            (
                RoFormerPfnForRelTokenizer,
                RoFormerPfnRelConfig,
                RoFormerForPfnRelExtraction,
            ),
        "bert-onerel-relation":
            (
                BertOneRelForRelTokenizer,
                BertOneRelRelConfig,
                BertForOneRelRelExtraction,
            ),
        "roformer-onerel-relation":
            (
                RoFormerOneRelForRelTokenizer,
                RoFormerOneRelRelConfig,
                RoFormerForOneRelRelExtraction,
            ),
    }
)

REL_TASK_NAMES = list(REL_MODEL_MAPPING.keys())


def load_rel_tokenizer(
    model_name_or_path: Union[str, os.PathLike],
    task_name: str = "bert-gplinker-relation",
) -> "PreTrainedTokenizer":
    assert task_name in REL_TASK_NAMES, f"Avialable task names are {REL_TASK_NAMES}"
    tokenizer_cls, _, _ = REL_MODEL_MAPPING[task_name]
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    tokenizer.__class__.register_for_auto_class()
    return tokenizer


def get_rel_data_collator(
    num_labels: int, tokenizer: "PreTrainedTokenizer",
) -> Callable:
    try:
        data_collator = tokenizer.get_collate_fn(num_labels)
    except TypeError:
        data_collator = tokenizer.get_collate_fn()
    return data_collator


def load_rel_model(
    task_name: str,
    model_name_or_path: Union[str, os.PathLike],
    schemas: List[Any],
    _fast_init: bool = False,
    **model_config_kwargs: Any,
) -> "PreTrainedModel":
    kwargs = {"schemas": schemas}
    kwargs.update(model_config_kwargs)

    _, config_cls, model_cls = REL_MODEL_MAPPING[task_name]
    config, unused_kwargs = config_cls.from_pretrained(
        model_name_or_path, **kwargs, return_unused_kwargs=True,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)
    config.__class__.register_for_auto_class()

    model = model_cls.from_pretrained(model_name_or_path, config=config, _fast_init=_fast_init)
    model.__class__.register_for_auto_class()

    return model
