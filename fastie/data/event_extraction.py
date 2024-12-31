import json
import os
import random
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Optional,
    Tuple,
    TYPE_CHECKING,
    List,
    Any,
    Dict,
)

from datasets import load_dataset

from ..extras import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset

logger = get_logger(__name__)


def load_labels(schema_file: Union[str, Path]) -> List[str]:
    labels = []
    with open(schema_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            event_type = data["event_type"]
            roles = ["触发词"] + [d["role"] for d in data["role_list"]]
            labels.extend(f"{event_type}@{role}" for role in roles)
    return sorted(list(set(labels)))


def process_example(example: Dict[str, Any], label_column_name: str) -> Dict[str, Any]:
    events = []
    for event in example[label_column_name]:
        trigger = event["trigger"]
        offset1 = len(trigger) - len(trigger.lstrip())
        events.append([
            [
                event["event_type"],
                "触发词",
                trigger,
                str(event["trigger_start_index"] + offset1),
                str(event["trigger_start_index"] + offset1 + len(trigger.strip())),
            ]
        ])
        for argument in event["arguments"]:
            arg_text = argument["argument"]
            offset2 = len(arg_text) - len(arg_text.lstrip())
            events[-1].append([
                event["event_type"],
                argument["role"],
                arg_text,
                str(argument["argument_start_index"] + offset2),
                str(argument["argument_start_index"] + offset2 + len(arg_text.strip())),
            ])
    del example["event_list"]
    return {"target": events}


def log_dataset_samples(dataset: "Dataset", dataset_name: str) -> None:
    sample_index = random.randint(0, len(dataset) - 1)
    logger.info(f"Length of {dataset_name} set: {len(dataset)}")
    logger.info(f"Sample {sample_index} of the {dataset_name} set:")
    for key, value in dataset[sample_index].items():
        logger.info(f"{key} = {value}")


def load_ee_train_dev_dataset(
    schema_file: Union[str, Path],
    tokenizer: "PreTrainedTokenizer",
    dataset_dir: Union[str, Path],
    train_file: Union[str, Path],
    validation_file: Union[str, Path] = None,
    text_column_name: str = "text",
    label_column_name: str = "event_list",
    train_val_split: Optional[int] = None,
    train_max_length: Optional[int] = 256,
    val_max_length: Optional[int] = 256,
    num_workers: Optional[int] = None,
    is_chinese: Optional[bool] = True,
    shuffle_train_dataset: Optional[bool] = False,
    shuffle_seed: Optional[int] = 42,
) -> Tuple["Dataset", "Dataset", List[Any]]:
    labels = load_labels(schema_file)
    label2id = {l: int(i) for i, l in enumerate(labels)}
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

    process_fn = partial(process_example, label_column_name=label_column_name)
    train_dataset = dataset["train"].map(process_fn)
    val_dataset = dataset["validation"].map(process_fn)

    task_name = str(tokenizer.__class__.__name__)[:20]
    convert_to_features_train = partial(
        tokenizer.convert_to_features,
        max_length=train_max_length,
        label_to_id=label2id,
        text_column_name=text_column_name,
        label_column_name="target",
        is_chinese=is_chinese,
    )
    train_dataset = train_dataset.map(
        convert_to_features_train,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train datasets",
        new_fingerprint=f"train-{train_max_length}-{task_name}",
        num_proc=num_workers if num_workers else None,
    )

    convert_to_features_val = partial(
        tokenizer.convert_to_features,
        max_length=val_max_length,
        label_to_id=label2id,
        text_column_name=text_column_name,
        label_column_name="target",
        is_chinese=is_chinese,
        mode="validation",
    )
    val_dataset = val_dataset.map(
        convert_to_features_val,
        batched=True,
        desc="Running tokenizer on validation datasets",
        new_fingerprint=f"validation-{val_max_length}-{task_name}",
        num_proc=num_workers if num_workers else None,
    )

    log_dataset_samples(train_dataset, "training")
    log_dataset_samples(val_dataset, "validation")

    return train_dataset, val_dataset, labels
