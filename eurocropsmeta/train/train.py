import logging
import math
from collections.abc import Sequence
from typing import Any, Literal, Type, cast

import optuna
import torch
import torch.nn as nn
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from eurocropsmeta.dataset.task import Task
from eurocropsmeta.experiment.logger import MLFlowCallback
from eurocropsmeta.models.base import DataItem, Model
from eurocropsmeta.train.callback import TrainCallback
from eurocropsmeta.train.utils import EarlyStopping, TaskMetric
from eurocropsmeta.utils import BaseConfig

logger = logging.getLogger(__name__)


class TrainConfig(BaseConfig):
    """Configuration class for vanilla training.

    Args:
        batch_size: The number of samples per batch.
        epochs: The number of epochs.
        steps: The number of total training steps. Ignored if epochs is not 0.
        lr: Learning rate for Training algorithm
            If this is a triple, the entries are interpreted as the learning rates
            for the backbone and head part of the model.
        optimizer: PyTorch optimizer used for training
        stop_early: Whether to use early stopping.
            Default: False
        patience:  Number of validation periods the validation loss is
            allowed to decrease. This corresponds to the number of validation periods
            which can either be batches or epochs.
    """

    batch_size: int
    epochs: int = 0
    steps: int = Field(0, validate_default=True)
    head_lr: float | None = Field(None, validate_default=True)
    backbone_lr: float | None = Field(None, validate_default=True)
    lr: float | None = Field(None, validate_default=True)
    optimizer: Literal["SGD", "Adam", "AdamW"] = "Adam"
    weight_decay: float = 0.0
    cosine_annealing: int = 0

    stop_early: bool = False
    patience: int | None = None

    def hyperparameters(self) -> dict[str, str | int | float | None]:
        """Return training hyperparameters."""
        params: dict[str, Any] = {
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "cosine_annealing": self.cosine_annealing,
        }
        if self.head_lr is not None and self.backbone_lr is not None:
            params["backbone_lr"] = self.backbone_lr
            params["head_lr"] = self.head_lr
        else:
            params["lr"] = self.lr
        return params

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: int, info: ValidationInfo) -> int:
        """Ensure that either steps or epochs are non-zero."""
        if v == 0 and info.data["epochs"] == 0:
            raise ValueError("Either epochs or steps must be non-zero.")
        return v

    @field_validator("backbone_lr")
    @classmethod
    def validate_separate_lr(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Ensure that backbone_lr is specified jointly with head_lr."""
        valid_head_lr = "head_lr" in info.data and info.data["head_lr"] is not None
        if (v is None and valid_head_lr) or (v is not None and not valid_head_lr):
            logger.warning(
                "Both head_lr and backbone_lr need to be jointly specified. "
                "Going to ignore separate head_lr and backbone_lr."
            )
            return None
        return v

    @field_validator("lr")
    @classmethod
    def validate_all_lr(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Ensure that either head_lr and backbone_lr or lr are specified."""
        valid_head_and_backbone_lr = (
            "head_lr" in info.data
            and "backbone_lr" in info.data
            and info.data["head_lr"] is not None
            and info.data["backbone_lr"] is not None
        )
        if v is None and not valid_head_and_backbone_lr:
            raise ValueError("Either both head_lr and backbone_lr or lr must be specified.")
        elif v is not None and valid_head_and_backbone_lr:
            logger.warning(
                "Both head_lr and backbone_lr as well as lr are specified. "
                "Going to ignore lr and use separate head_lr and backbone_lr."
            )
            return None
        return v


class Trainer:
    """Class for running vanilla training algorithm.

    Args:
        config: Training config.
        callbacks: Callbacks to call during training
    """

    def __init__(self, config: TrainConfig, callbacks: list[TrainCallback] | None = None):
        self.config = config
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    @staticmethod
    def _build_optimizer(
        model: Model,
        optimizer: Literal["SGD", "Adam", "AdamW"],
        head_lr: float | None,
        backbone_lr: float | None,
        lr: float | None,
        weight_decay: float,
    ) -> Optimizer:
        optimizer_class: Type[SGD | Adam | AdamW] = {
            "SGD": SGD,
            "Adam": Adam,
            "AdamW": AdamW,
        }[optimizer]
        if head_lr is not None and backbone_lr is not None:
            parameters: list[dict] = [
                {
                    "params": model.backbone.parameters(),
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": model.head.parameters(),
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                },
            ]
            if model.encoder is not None:
                parameters.append(
                    {
                        "params": model.encoder.parameters(),
                        "lr": backbone_lr,
                        "weight_decay": weight_decay,
                    }
                )
            return optimizer_class(
                parameters,
                lr=backbone_lr,
            )
        else:  # lr cannot be None as well
            return optimizer_class(
                model.parameters(), lr=cast(float, lr), weight_decay=weight_decay
            )

    def _build_scheduler(
        self,
        optimizer: Optimizer,
        steps_per_epoch: int,
    ) -> tuple[CosineAnnealingWarmRestarts | None, int | None]:
        max_num_steps = steps_per_epoch * self.config.epochs
        if self.config.cosine_annealing == 0:
            return None, self.config.patience
        elif self.config.cosine_annealing == 1:
            t_0 = max_num_steps + 1
        else:
            t_0 = math.floor(max_num_steps / self.config.cosine_annealing)
        return CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=0.0), math.floor(
            self.config.epochs / self.config.cosine_annealing
        )

    def train(self, model: Model, task: Task, validate_every: int = 1) -> Model:
        """Train model with configured optimizer and scheduling.

        Args:
            model: Model to adapt.
            task: Task that contains a train, validation and test dataset.
            validate_every: Interval between validation runs, either corresponding to the
                number of epochs (if epoch>0) or to the number of training batches (if epoch==0).

        Returns:
            Adapted model
        """

        # setting encoder to eval mode (for TIML)
        if model.encoder is not None:
            model.encoder.eval()

        train_dl = task.train_dl(batch_size=self.config.batch_size)
        val_dl = task.val_dl(batch_size=self.config.batch_size)

        optimizer = self._build_optimizer(
            model=model,
            optimizer=self.config.optimizer,
            head_lr=self.config.head_lr,
            backbone_lr=self.config.backbone_lr,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler, patience = self._build_scheduler(optimizer, steps_per_epoch=len(train_dl))

        # initialize the early_stopping object
        early_stopping: EarlyStopping | None = None
        if self.config.stop_early:
            early_stopping = EarlyStopping(patience=patience)

        # If epoch>0 we count by epochs.
        if self.config.epochs:
            epochs_bar = trange(self.config.epochs, desc="Epochs", position=0, leave=True)
            step = 0
            for epoch in epochs_bar:
                for _, batch in enumerate(tqdm(train_dl, desc="Batches")):
                    self._inner_step(
                        model=model,
                        train_batch=batch,
                        task=task,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                    )
                    step += 1
                validate = (epoch + 1) % validate_every == 0
                if validate:
                    stopped = self._validate(model, task, early_stopping, step, val_dl)
                    if stopped:
                        break

        # Epochs is zero, so count by steps.
        else:
            batches_bar = trange(
                min(len(train_dl), self.config.steps),
                desc="Batches",
                position=0,
                leave=True,
            )
            for step, batch in zip(batches_bar, train_dl):
                self._inner_step(
                    model=model,
                    train_batch=batch,
                    task=task,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                )
                validate = (step + 1) % validate_every == 0
                if validate:
                    stopped = self._validate(model, task, early_stopping, step, val_dl)
                    if stopped:
                        break

        for callback in self.callbacks:
            callback.end_callback(model)
        return model

    def _inner_step(
        self,
        model: Model,
        train_batch: tuple[DataItem, torch.Tensor],
        task: Task,
        optimizer: Optimizer,
        scheduler: CosineAnnealingWarmRestarts | None,
        step: int,
    ) -> None:
        train_loss = _train_step(
            model=model,
            batch=train_batch,
            loss_fn=task.loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        train_metrics: Sequence[TaskMetric] = []
        for callback in self.callbacks:
            if step % 10 == 0:
                if isinstance(callback, MLFlowCallback) and scheduler:
                    cast(MLFlowCallback, callback).mlflow_logger.log(
                        "lr", scheduler.get_last_lr()[0], global_step=step
                    )
                callback.train_callback(train_loss, train_metrics, model=model, step=step)

    def _validate(
        self,
        model: Model,
        task: Task,
        early_stopping: EarlyStopping | None,
        step: int,
        val_dl: DataLoader,
    ) -> bool:
        val_loss, val_metrics = self._evaluation_loop(val_dl, task, model)

        for callback in self.callbacks:
            try:
                callback.validation_callback(
                    val_loss,
                    val_metrics,
                    model,
                    step=step,
                )
            except optuna.TrialPruned as e:
                # even if we prune, make sure to properly end all callbacks
                for callback in self.callbacks:
                    callback.end_callback(model)
                raise e  # reraise the pruning exception after ending all callbacks

        stopped = False
        if early_stopping:
            stopped = early_stopping(val_loss, model)

        return stopped

    def test(
        self,
        model: Model,
        task: Task,
    ) -> tuple[float, Sequence[TaskMetric]]:
        """Test trained model on test set.

        Args:
            model: Trained Model.
            task: Task dataset to use.

        Returns:
            Average test loss per batch
            Test Metrics
        """

        test_dl = task.test_dl(batch_size=self.config.batch_size)

        test_loss, test_metrics = self._evaluation_loop(test_dl, task, model)

        for callback in self.callbacks:
            callback.test_callback(
                test_loss,
                test_metrics,
                model,
                step=None,
            )

        return test_loss, test_metrics

    def _evaluation_loop(
        self, dataloader: DataLoader, task: Task, model: Model
    ) -> tuple[float, Sequence[TaskMetric]]:
        """Returns loss, dict of scalar metrics, dict of additional (tensor) metrics."""
        for metric in task.metrics:
            metric.reset()

        eval_loss = 0.0
        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                loss = _eval_step(
                    model,
                    batch,
                    task.loss_fn,
                    task.metrics,
                )

            eval_loss += loss
        eval_loss = eval_loss / len(dataloader)

        return eval_loss, task.metrics


def _train_step(
    model: Model,
    batch: tuple[DataItem, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: CosineAnnealingWarmRestarts | None = None,
) -> float:
    optimizer.zero_grad()
    model.train()

    data, target = batch[0].to(model.device), batch[1].to(model.device)

    out = model(data)
    train_loss = loss_fn(out, target).to(model.device)

    train_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss = train_loss.detach().cpu().item()

    return cast(float, loss)


def _eval_step(
    model: Model,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: nn.Module,
    metrics: Sequence[TaskMetric],
) -> float:
    model.eval()

    data, target = batch[0].to(model.device), batch[1].to(model.device)

    out = model(data)
    test_loss = loss_fn(out, target)

    for metric in metrics:
        metric.update(out.detach().cpu(), target.detach().cpu())

    return cast(float, test_loss.detach().cpu().item())
