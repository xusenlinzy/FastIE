import os
import random
from pathlib import Path
from typing import (
    Union,
    Optional,
    Tuple,
    TYPE_CHECKING,
    List,
    Any,
)

from datasets import load_dataset, ClassLabel

from ..extras import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset

logger = get_logger(__name__)


def load_cls_train_dev_dataset(
    tokenizer: "PreTrainedTokenizer",
    dataset_dir: Union[str, Path],
    train_file: Union[str, Path],
    validation_file: Union[str, Path] = None,
    text_column_name: str = "text",
    label_column_name: str = "label",
    train_val_split: Optional[int] = None,
    with_indices: Optional[bool] = False,
    train_max_length: Optional[int] = 256,
    val_max_length: Optional[int] = 256,
    num_workers: Optional[int] = None,
    is_chinese: Optional[bool] = True,
    shuffle_train_dataset: Optional[bool] = False,
    shuffle_seed: Optional[int] = 42,
) -> Tuple["Dataset", "Dataset", List[Any]]:
    data_files = dict()
    if train_file:
        data_files["train"] = os.path.join(dataset_dir, train_file)
    if validation_file:
        data_files["validation"] = os.path.join(dataset_dir, validation_file)

    extension = train_file.split(".")[-1]
    dataset = load_dataset(
        extension, data_files=data_files, cache_dir=dataset_dir,
    )
    if train_val_split is not None:
        split = dataset["train"].train_test_split(train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    if shuffle_train_dataset:
        logger.info("Shuffling the training dataset")
        dataset["train"] = dataset["train"].shuffle(seed=shuffle_seed)

    input_feature_fields = [
        k for k, v in dataset["train"].features.items() if k not in ["label", "idx", "id"]
    ]

    def convert_to_features(example_batch):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[input_feature_fields[0]], example_batch[input_feature_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]
        # Tokenize the text/text pairs
        return tokenizer(
            texts_or_text_pairs,
            padding=False,
            truncation=True,
            max_length=train_max_length,
        )

    dataset = dataset.map(
        convert_to_features,
        batched=True,
        num_proc=num_workers if num_workers else None,
        desc="Running tokenizer on dataset",
    )
    dataset = dataset.rename_column("label", "labels")

    cols_to_keep = [
        x for x in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        if x in dataset["train"].features
    ]
    if not isinstance(dataset["train"].features["labels"], ClassLabel):
        dataset = dataset.class_encode_column("labels")
    dataset.set_format("torch", columns=cols_to_keep)
    labels = dataset["train"].features["labels"]

    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Length of training set: {len(train_dataset)}")
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    for index in random.sample(range(len(val_dataset)), 1):
        logger.info(f"Length of validation set: {len(val_dataset)}")
        logger.info(f"Sample {index} of the validation set:")
        for k, v in val_dataset[index].items():
            logger.info(f"{k} = {v}")

    return train_dataset, val_dataset, labels.names
