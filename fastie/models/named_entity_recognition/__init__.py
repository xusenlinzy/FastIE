import os
from collections import OrderedDict
from typing import (
    Union,
    Any,
    List,
    TYPE_CHECKING,
    Callable,
)

from .cnn import *
from .crf import *
from .global_pointer import *
from .span import *
from .tplinker import *
from .w2ner import *

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


NER_MODEL_MAPPING = OrderedDict(
    {
        "bert-crf-ner":
            (
                BertCrfNerTokenizer,
                BertCrfNerConfig,
                BertForCrfNer,
            ),
        "roformer-crf-ner":
            (
                RoFormerCrfNerTokenizer,
                RoFormerCrfNerConfig,
                RoFormerForCrfNer,
            ),
        "bert-cascade-crf-ner":
            (
                BertCascadeCrfNerTokenizer,
                BertCascadeCrfNerConfig,
                BertForGlobalPointerNer,
            ),
        "roformer-cascade-crf-ner":
            (
                RoFormerCascadeCrfNerTokenizer,
                RoFormerCascadeCrfNerConfig,
                RoFormerForGlobalPointerNer,
            ),
        "bert-gp-ner":
            (
                BertGlobalPointerForNerTokenizer,
                BertGlobalPointerNerConfig,
                BertForGlobalPointerNer,
            ),
        "roformer-gp-ner":
            (
                RoFormerGlobalPointerForNerTokenizer,
                RoFormerGlobalPointerNerConfig,
                RoFormerForGlobalPointerNer,
            ),
        "bert-tplinker-ner":
            (
                BertTPLinkerForNerTokenizer,
                BertTPLinkerNerConfig,
                BertForTPLinkerNer,
            ),
        "roformer-tplinker-ner":
            (
                RoFormerTPLinkerForNerTokenizer,
                RoFormerTPLinkerNerConfig,
                RoFormerForTPLinkerNer,
            ),
        "bert-span-ner":
            (
                BertSpanNerTokenizer,
                BertSpanNerConfig,
                BertForSpanNer,
            ),
        "roformer-span-ner":
            (
                RoFormerSpanNerTokenizer,
                RoFormerSpanNerConfig,
                RoFormerForSpanNer,
            ),
        "bert-w2ner-ner":
            (
                BertW2nerTokenizer,
                BertW2nerConfig,
                BertForW2ner,
            ),
        "roformer-w2ner-ner":
            (
                RoFormerW2nerTokenizer,
                RoFormerW2nerConfig,
                RoFormerForW2ner,
            ),
        "bert-cnn-ner":
            (
                BertCnnNerTokenizer,
                BertCnnNerConfig,
                BertForCnnNer,
            ),
        "roformer-cnn-ner":
            (
                RoFormerCnnNerTokenizer,
                RoFormerCnnNerConfig,
                RoFormerForCnnNer,
            ),
    }
)

NER_TASK_NAMES = list(NER_MODEL_MAPPING.keys())


def load_ner_tokenizer(
    model_name_or_path: Union[str, os.PathLike],
    task_name: str = "bert-gp-ner",
) -> "PreTrainedTokenizer":
    assert task_name in NER_TASK_NAMES, f"Avialable task names are {NER_TASK_NAMES}"
    tokenizer_cls, _, _ = NER_MODEL_MAPPING[task_name]
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path)
    tokenizer.__class__.register_for_auto_class()
    return tokenizer


def get_ner_data_collator(
    num_labels: int, tokenizer: "PreTrainedTokenizer",
) -> Callable:
    try:
        data_collator = tokenizer.get_collate_fn(num_labels)
    except TypeError:
        data_collator = tokenizer.get_collate_fn()
    return data_collator


def load_ner_model(
    task_name: str,
    model_name_or_path: Union[str, os.PathLike],
    schemas: List[Any],
    _fast_init: bool = False,
    **model_config_kwargs: Any,
) -> "PreTrainedModel":
    kwargs = {"schemas": schemas}
    kwargs.update(model_config_kwargs)

    _, config_cls, model_cls = NER_MODEL_MAPPING[task_name]
    config, unused_kwargs = config_cls.from_pretrained(
        model_name_or_path, **kwargs, return_unused_kwargs=True,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)
    config.__class__.register_for_auto_class()

    model = model_cls.from_pretrained(model_name_or_path, config=config, _fast_init=_fast_init)
    model.__class__.register_for_auto_class()

    return model
