from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    task_name: str = field(
        default=None,
        metadata={
            "help": "Task name for load tokenizer and model."
        },
    )
    dataset_dir: str = field(
        default="data",
        metadata={
            "help": "Path to the folder containing the datasets."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A json file containing the training data."
        }
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A json file containing the validation data."
        }
    )
    schema_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A json file containing the schema."
        }
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={
            "help": "The name of the text column in the input dataset or a JSON file. "
        },
    )
    label_column_name: Optional[str] = field(
        default="entities",
        metadata={
            "help": "The name of the label column in the input dataset or a JSON file. "
        },
    )
    train_max_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length of train dataset after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    validation_max_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length of valid dataset after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle the train dataset or not."
        }
    )
    shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be used to shuffle the train dataset."
        }
    )
    is_chinese: bool = field(
        default=True,
        metadata={
            "help": "Whether the language is Chinese."
        }
    )
    validation_split_percentage: Optional[int] = field(
        default=None,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
