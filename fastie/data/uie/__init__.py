import os
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Tuple,
    Optional,
    TYPE_CHECKING
)

import numpy as np
from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset


def load_uie_train_dev_dataset(
    tokenizer: "PreTrainedTokenizer",
    dataset_dir: Union[str, Path],
    train_file: Union[str, Path],
    validation_file: Union[str, Path] = None,
    train_val_split: Optional[int] = None,
    train_max_length: Optional[int] = 256,
    num_workers: Optional[int] = None,
    shuffle_train_dataset: Optional[bool] = False,
    shuffle_seed: Optional[int] = 42,
) -> Tuple["Dataset", "Dataset"]:
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
        dataset["train"] = dataset["train"].shuffle(seed=shuffle_seed)

    convert_to_features = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=train_max_length,
    )
    dataset = dataset.map(
        convert_to_features,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on train datasets",
        num_proc=num_workers if num_workers else None,
    )

    train_dataset, val_dataset = dataset["train"], dataset["validation"]
    return train_dataset, val_dataset, None


def convert_example(example: dict, tokenizer: "PreTrainedTokenizer", max_seq_len: int):
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"][0]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue

        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias

    start_ids = np.zeros((max_seq_len,))
    end_ids = np.zeros((max_seq_len,))

    def map_offset(ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        return next((index for index, span in enumerate(offset_mapping) if span[0] <= ori_offset < span[1]), -1)

    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    input_ids = np.array(encoded_inputs["input_ids"][0], dtype="int64")
    attention_mask = np.asarray(encoded_inputs["attention_mask"][0], dtype="int64")
    token_type_ids = np.asarray(encoded_inputs["token_type_ids"][0], dtype="int64")

    tokenized_output = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "start_positions": start_ids,
        "end_positions": end_ids,
    }

    tokenized_output = {
        k: np.pad(v, (0, max_seq_len - v.shape[-1]), 'constant')
        for k, v in tokenized_output.items()
    }
    return tokenized_output
