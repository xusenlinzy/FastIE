from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    device: str = field(
        default="cpu",
        metadata={
            "help": "The device to load the model for inference."
        },
    )
