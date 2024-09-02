import os
import sys

from transformers import HfArgumentParser

from fastie.hparams import (
    DataArguments,
    ModelArguments,
    FinetuneArguments,
)
from fastie.train.tuner import run_task


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, FinetuneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    run_task(data_args, model_args, training_args, use_mdp=True)


if __name__ == "__main__":
    main()
