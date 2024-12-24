from typing import (
    Union,
    Optional,
    List,
    Dict,
    NamedTuple,
    Set,
    Any,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_apex_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_xpu_available,
    is_torch_npu_available,
)

from ..extras import get_logger, FGM
from ..metrics.extraction import (
    ExtractionScore,
    EventExtractionScore,
    SpanEvaluator,
)

if is_apex_available():
    from apex import amp

logger = get_logger(__name__)


class EvalLoopOutput(NamedTuple):
    predictions: List[Any]
    groundtruths: List[Any]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class BaseTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None
        self.do_adv = getattr(self.args, "do_adv", False)
        if self.do_adv:
            self.fgm = FGM(self.model)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if self.do_adv and hasattr(self, "fgm"):
            self.fgm.attack()

            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps

            if self.use_apex:
                with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss_adv)

            self.fgm.restore()

        del inputs
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps

        return loss.detach()


class OptimizerTrainer(BaseTrainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            differential_learning_rate = getattr(self.args, "differential_learning_rate", None)
            if differential_learning_rate is not None:
                try:
                    base_model = getattr(opt_model, opt_model.base_model_prefix)
                except AttributeError:
                    base_model = getattr(opt_model, "encoder")

                base_model_params = list(base_model.named_parameters())
                base_model_param_ids = [id(p) for n, p in base_model_params]
                differential_model_param = [
                    (n, p) for n, p in opt_model.named_parameters()
                    if id(p) not in base_model_param_ids
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in base_model_params if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in base_model_params if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in differential_model_param if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": differential_learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in differential_model_param if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": differential_learning_rate,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_model_outputs(self, model, inputs, ignore_keys, args):
        _, predictions, groundtruths = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys)
        self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        self.metrics.update(groundtruths, predictions)
        return predictions, groundtruths

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        all_predictions, all_groundtruths = [], []
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            predictions, groundtruths = self.get_model_outputs(model, inputs, ignore_keys, args)
            all_predictions.extend(predictions)
            all_groundtruths.extend(groundtruths)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore, we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        metrics = self.metrics.value()
        self.metrics.reset()
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_predictions,
            groundtruths=all_groundtruths,
            metrics=metrics,
            num_samples=num_samples,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[Set]], Optional[List[Set]]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            outputs = model(**inputs)
            predictions, groundtruths = outputs.predictions, outputs.groundtruths
        if prediction_loss_only:
            return loss, None, None
        return loss, predictions, groundtruths

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return eval_dataloader


class ExtractionTrainer(OptimizerTrainer):
    def __init__(self, event: bool = False, **kwargs):
        super().__init__(**kwargs)
        if event:
            self.metrics = EventExtractionScore()
        else:
            self.metrics = ExtractionScore()


class UIETrainer(OptimizerTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = SpanEvaluator()

    def get_model_outputs(self, model, inputs, ignore_keys, args):
        loss, start_prob, end_prob, start_ids, end_ids = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys)
        self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        num_correct, num_infer, num_label = self.metrics.compute(
            start_prob, end_prob, start_ids, end_ids,
        )
        self.metrics.update(num_correct, num_infer, num_label)
        return (start_prob, end_prob), (start_ids, end_ids)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        start_positions = inputs["start_positions"].numpy()
        end_positions = inputs["end_positions"].numpy()
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            outputs = model(**inputs)
            start_prob = outputs.start_prob.detach().cpu().numpy()
            end_prob = outputs.end_prob.detach().cpu().numpy()
        if prediction_loss_only:
            return loss, None, None
        return loss, start_prob, end_prob, start_positions, end_positions


def compute_classification_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    except:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
