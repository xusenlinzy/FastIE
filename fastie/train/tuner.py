import os
from collections import OrderedDict
from typing import (
    Any,
    TYPE_CHECKING,
    Optional,
    List,
    Dict,
)

from transformers import default_data_collator

from .callbacks import LogCallback
from .trainer import (
    ExtractionTrainer,
    UIETrainer,
    compute_classification_metrics,
    BaseTrainer,
)
from ..data import (
    load_ner_train_dev_dataset,
    load_rel_train_dev_dataset,
    load_ee_train_dev_dataset,
    load_uie_train_dev_dataset,
    load_cls_train_dev_dataset,
)
from ..extras import get_logger
from ..hparams.parser import get_train_args
from ..models.event_extraction import (
    load_event_tokenizer,
    load_event_model,
    get_event_data_collator,
)
from ..models.named_entity_recognition import (
    load_ner_tokenizer,
    load_ner_model,
    get_ner_data_collator,
)
from ..models.relation_extraction import (
    load_rel_model,
    load_rel_tokenizer,
    get_rel_data_collator,
)
from ..models.text_classification import (
    load_cls_tokenizer,
    load_cls_model,
    get_cls_data_collator,
)
from ..models.uie import load_uie_model, load_uie_tokenizer

if TYPE_CHECKING:
    from ..hparams import (
        DataArguments,
        ModelArguments,
        FinetuneArguments,
    )
    from transformers import TrainerCallback

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

logger = get_logger(__name__)


LOAD_FUNCTION_MAPPING = OrderedDict(
    {
        "ner": (
            load_ner_train_dev_dataset,
            load_ner_tokenizer,
            load_ner_model,
            get_ner_data_collator,
        ),
        "relation": (
            load_rel_train_dev_dataset,
            load_rel_tokenizer,
            load_rel_model,
            get_rel_data_collator,
        ),
        "event": (
            load_ee_train_dev_dataset,
            load_event_tokenizer,
            load_event_model,
            get_event_data_collator,
        ),
        "uie": (
            load_uie_train_dev_dataset,
            load_uie_tokenizer,
            load_uie_model,
            None,
        ),
        "cls": (
            load_cls_train_dev_dataset,
            load_cls_tokenizer,
            load_cls_model,
            get_cls_data_collator,
        ),
    }
)


def run_task(
    data_args: "DataArguments",
    model_args: "ModelArguments",
    finetune_args: "FinetuneArguments",
    callbacks: Optional[List["TrainerCallback"]] = [],
    **model_config_kwargs: Any,
):
    callbacks.append(LogCallback())

    task_name = data_args.task_name
    try:
        task_key = task_name.strip().split("-")[-1]
        load_datasets, load_tokenizer, load_model, load_collator = LOAD_FUNCTION_MAPPING[task_key]
    except KeyError:
        raise ValueError(f"Not supported task name {task_name}")

    tokenizer = load_tokenizer(model_args.model_name_or_path, task_name)
    data_kwargs = dict(
        tokenizer=tokenizer,
        dataset_dir=data_args.dataset_dir,
        train_file=data_args.train_file,
        validation_file=data_args.validation_file,
        train_val_split=data_args.validation_split_percentage,
        train_max_length=data_args.train_max_length,
        num_workers=data_args.preprocessing_num_workers,
        shuffle_seed=data_args.shuffle_seed,
        shuffle_train_dataset=data_args.shuffle_train_dataset,
    )
    if task_name != "uie":
        data_kwargs.update(
            dict(
                text_column_name=data_args.text_column_name,
                label_column_name=data_args.label_column_name,
                val_max_length=data_args.validation_max_length,
                is_chinese=data_args.is_chinese,
            )
        )
    if task_name.endswith("event"):
        data_kwargs["schema_file"] = data_args.schema_file
    train_dataset, eval_dataset, schemas = load_datasets(**data_kwargs)

    if task_name == "uie":
        model = load_model(task_name, model_args.model_name_or_path)
    else:
        model = load_model(
            task_name,
            model_args.model_name_or_path,
            schemas=schemas,
            **model_config_kwargs,
        )

    data_collator = load_collator(len(schemas), tokenizer) if load_collator is not None else default_data_collator
    finetune_args.remove_unused_columns = False
    finetune_args.save_safetensors = False  # TODO: fix removing shared tensors

    trainer_cls = ExtractionTrainer
    trainer_kwargs = {}
    if task_name == "uie":
        trainer_cls = UIETrainer
    elif task_name.endswith("cls"):
        trainer_cls = BaseTrainer
        trainer_kwargs["compute_metrics"] = compute_classification_metrics
    elif task_name.endswith("event"):
        trainer_kwargs["event"] = True

    trainer = trainer_cls(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **trainer_kwargs
    )

    if finetune_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=finetune_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if finetune_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def run_exp(args: Optional[Dict[str, Any]] = None) -> None:
    data_args, model_args, training_args = get_train_args(args)
    run_task(data_args, model_args, training_args)
