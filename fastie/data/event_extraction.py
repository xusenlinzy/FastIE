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
)

from datasets import load_dataset

from ..extras import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset

logger = get_logger(__name__)


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
    labels = []
    with open(schema_file, "r") as f:
        for l in f:
            l = json.loads(l)
            t = l["event_type"]
            for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
                labels.append(f"{t}@{r}")
    labels = sorted(labels)
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

    def duee_v1_process(example):
        events = []
        for e in example[label_column_name]:
            offset1 = len(e["trigger"]) - len(e["trigger"].lstrip())
            events.append(
                [
                    [
                        e["event_type"],
                        "触发词",
                        e["trigger"],
                        str(e["trigger_start_index"] + offset1),
                        str(e["trigger_start_index"] + offset1 + len(e["trigger"].strip())),
                    ]
                ]
            )
            for a in e["arguments"]:
                offset2 = len(a["argument"]) - len(a["argument"].lstrip())
                events[-1].append(
                    [
                        e["event_type"],
                        a["role"],
                        a["argument"],
                        str(a["argument_start_index"] + offset2),
                        str(a["argument_start_index"] + offset2 + len(a["argument"].strip())),
                    ]
                )
        del example["event_list"]
        return {"target": events}

    train_dataset = dataset["train"].map(duee_v1_process)
    val_dataset = dataset["validation"].map(duee_v1_process)

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

    return train_dataset, val_dataset, labels
