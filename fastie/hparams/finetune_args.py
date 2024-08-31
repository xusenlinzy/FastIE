from dataclasses import field, dataclass

from transformers import TrainingArguments


@dataclass
class FinetuneArguments(TrainingArguments):
    differential_learning_rate: float = field(
        default=None,
        metadata={"help": "The differential learning rate for AdamW."}
    )
    do_adv: bool = field(
        default=False,
        metadata={
            "help": "Whether to run adversarial training."
        }
    )
