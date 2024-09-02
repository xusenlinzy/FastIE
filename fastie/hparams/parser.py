import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import transformers
from transformers import HfArgumentParser

from ..extras.logging import get_logger
from ..hparams import (
    DataArguments,
    ModelArguments,
    FinetuneArguments,
    InferArguments,
)

logger = get_logger(__name__)


_TRAIN_ARGS = [DataArguments, ModelArguments, FinetuneArguments]
_TRAIN_CLS = Tuple[DataArguments, ModelArguments, FinetuneArguments]
_INFER_ARGS = [ModelArguments, InferArguments]
_INFER_CLS = Tuple[ModelArguments, InferArguments]


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return *parsed_args,


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    data_args, model_args, training_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    if training_args.do_train and data_args.dataset_dir is None:
        raise ValueError("Please specify dataset for training.")

    transformers.set_seed(training_args.seed)

    return data_args, model_args, training_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, infer_args = _parse_infer_args(args)
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"API server arguments: {infer_args}")
    return model_args, infer_args
