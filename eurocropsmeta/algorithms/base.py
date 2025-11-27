import logging
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from itertools import islice
from typing import Generic, Literal, TypeVar

import torch
from eurocropsml.dataset.base import DataItem
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from eurocropsmeta.algorithms.inner_loop import InnerLoopLearner
from eurocropsmeta.algorithms.utils import clone_module
from eurocropsmeta.dataset.task import Task
from eurocropsmeta.models.base import Model, ModelBuilder
from eurocropsmeta.settings import Settings
from eurocropsmeta.train.utils import TaskMetric
from eurocropsmeta.utils import BaseConfig

logger = logging.getLogger(__name__)


class MetaLearnAlgorithmConfig(BaseConfig):
    """Config for a meta-learning algorithm.

    Args:
        name: Name of the algorithm
        tasks_per_batch: The number of tasks sampled in each batch during task adaption.
        total_task_batches: The total number of task batches seen during training
        train_adaption_steps: Number of adaption steps in inner loop for train tasks.
        test_adaption_steps: Number of adaption steps in inner loop for test tasks.
            If this is None, fallback to train_adaption_steps.
        num_classes: Number of classes in classification head of the model.
            If this is None, then `reset_head` must be True.
        reset_head: Whether to reset the model head for each new task.
        outer_lr: Learning rate used in outer step.
        outer_weight_decay: Weight decay used in outer step.
        inner_optimizer: Name of the optimizer used in the inner task adaptation loop.
        outer_optimizer: Name of the optimizer used in the outer meta-learning loop.
        init_inner_momentum_from_outer: Initialize momentum buffers of the inner loop
            optimizer with the momentum buffers of the outer loop optimizer. This is
            disabled by default for general meta-learning algorithms and only takes
            effect for MAML if the inner and and outer loop both use Adam optimizers.
    """

    name: str
    tasks_per_batch: int
    total_task_batches: int

    train_adaption_steps: int
    test_adaption_steps: int | None = None

    num_classes: int | None = None
    reset_head: bool = True

    outer_lr: float
    outer_weight_decay: float = 0.0

    prefetch_data: bool = True

    inner_optimizer: Literal["SGD", "Adam"] = "SGD"
    outer_optimizer: Literal["SGD", "Adam", "AdamW"] = "Adam"
    init_inner_momentum_from_outer: bool = False

    def hyperparameters(self) -> dict[str, str | int | float | None]:
        """Return algorithm hyperparameters as dictionary."""
        return self.model_dump()

    @field_validator("reset_head")
    @classmethod
    def ensure_reset_head(cls, v: bool, info: ValidationInfo) -> bool:
        num_classes: int | None = info.data["num_classes"]
        if num_classes is None and not v:
            raise ValueError("Option 'reset_head' must be True, if 'num_classes' is None.")
        return v

    @field_validator("init_inner_momentum_from_outer")
    @classmethod
    def init_momentum_only_for_maml(cls, v: bool, info: ValidationInfo) -> bool:
        if v:
            logger.warning(
                "Initializing the exponentially running average momentum buffers "
                "of the inner loop optimizer from the buffers of the outer loop "
                "optimizer is only possible for the MAML algorithm and if "
                "Adam is used for both loops. This is not the case for the "
                "current configuration. Therefore, the chosen buffer initialization "
                "strategy will be ignored."
            )
        return False


MetaLearnAlgorithmConfigT = TypeVar("MetaLearnAlgorithmConfigT", bound=MetaLearnAlgorithmConfig)


class MetaLearnAlgorithm(Generic[MetaLearnAlgorithmConfigT]):
    """Optimization based meta learning algorithm.

    Args:
        config: Algorithm config.
        model_builder: Builder to use to create initial model.
    """

    def __init__(
        self,
        config: MetaLearnAlgorithmConfigT,
        model_builder: ModelBuilder,
    ):
        self.config = config
        num_classes = self.config.num_classes or 1

        self.model_builder = model_builder

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model_builder.build_classification_model(num_classes, device)
        self.inner_learner = self.build_inner_optimizer().to(device)
        self._outer_optimizer = self.build_outer_optimizer()

    @abstractmethod
    def outer_parameters(self) -> list[nn.Parameter]:
        pass

    def build_encoder_optimizer(self) -> Optimizer:
        raise NotImplementedError

    def encoder_parameters(self) -> list[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def build_inner_optimizer(self) -> InnerLoopLearner:
        """Build optimizer for inner loop."""

    def build_outer_optimizer(self) -> Optimizer:
        """Build optimizer for outer loop."""
        model_params = self.outer_parameters()
        params: list[dict[str, Iterable[nn.Parameter] | float | None]] = [
            {
                "params": model_params,
                "lr": self.config.outer_lr,
                "weight_decay": self.config.outer_weight_decay,
            }
        ]
        if inner_learner_params := self.inner_learner.optimizer_params():
            params += [inner_learner_params]

        if self.config.outer_weight_decay > 0 and self.config.outer_optimizer == "Adam":
            logger.warn(
                "Adam was specified as the optimizer in combination with "
                "weight decay for the outer loop. It is recommended to "
                "use AdamW instead."
            )
        match self.config.outer_optimizer:
            case "SGD":
                return SGD(
                    params, lr=0.01
                )  # lr required by SGD but overwritten for each params groups
            case "Adam":
                return Adam(params)
            case "AdamW":
                return AdamW(params)

    def inner_update(self, model: Model, train_loss: torch.Tensor, inner_step: int) -> None:
        """Perform inner loop step for given model and training loss."""
        pass

    def build_task_model(self, task: Task, context: Literal["adapt", "eval"]) -> Model:
        """Build model for training on given task."""
        task_model = clone_module(self.model) if context == "adapt" else deepcopy(self.model)
        if self.config.reset_head:
            self.model_builder.reset_head(task_model, task.num_classes)
        return task_model

    def adapt(
        self,
        tasks: list[Task],
    ) -> float:
        """Adapt model using given tasks."""
        outer_loss = torch.tensor(0.0, device=self.model.device)

        self._outer_optimizer.zero_grad()
        if hasattr(self, "_encoder_optimizer"):
            self._encoder_optimizer.zero_grad()

        for task in tasks:
            task_model = self.build_task_model(task, context="adapt")
            test_loss = self._inner_loop(model=task_model, task=task, context="adapt")
            outer_loss += test_loss
        outer_loss.div_(len(tasks))
        outer_loss.backward()
        self._outer_optimizer.step()
        if hasattr(self, "_encoder_optimizer"):
            self._encoder_optimizer.step()

        return outer_loss.item()

    def eval(self, task: Task) -> tuple[float, Sequence[TaskMetric], Model]:
        """Evaluate adapted model on given tasks.

        Args:
            task: Task to evaluate.

        Returns:
            Tuple of
            - Model loss on the task test set.
            - Dictionary of task metrics evaluated on task test set.
            - Task-specific version of model
        """
        for metric in task.metrics:
            metric.reset()
        task_model = self.build_task_model(task, context="eval")
        test_loss = self._inner_loop(model=task_model, task=task, context="eval")

        return test_loss.item(), task.metrics, task_model

    def _inner_loop(
        self,
        model: Model,
        task: Task,
        context: Literal["adapt", "eval"],
    ) -> torch.Tensor:
        if self.config.init_inner_momentum_from_outer:
            optimizer_state = self._outer_optimizer.state_dict()
            self.inner_learner.reset_optimizer_state(
                reset_values={
                    "exp_avgs": [
                        p["exp_avg"].clone().detach() for p in optimizer_state["state"].values()
                    ],
                    "exp_avg_sqs": [
                        p["exp_avg_sq"].clone().detach() for p in optimizer_state["state"].values()
                    ],
                }
            )
        else:
            self.inner_learner.reset_optimizer_state(reset_values=None)

        model.train()

        adaption_steps = self.config.train_adaption_steps
        if context == "eval" and self.config.test_adaption_steps is not None:
            adaption_steps = self.config.test_adaption_steps

        train_dl = task.train_dl(num_batches=adaption_steps)
        test_dl = task.test_dl(num_batches=adaption_steps)

        if len(train_dl) < adaption_steps:
            logger.warning(
                f"Trying to obtain {adaption_steps} batch(es) from a "
                f"dataloader that contains only {len(train_dl)} batch(es)."
                f"This might result in an unintended behavior."
            )
        for n, train_item in enumerate(islice(train_dl, adaption_steps)):
            train_data, train_target = train_item
            train_loss = _eval_batch(
                model=model,
                data=train_data,
                target=train_target,
                loss_fn=task.loss_fn,
            )

            model.zero_grad()
            self.inner_update(model, train_loss, inner_step=n)

        model.eval()
        test_loss = torch.tensor(0.0, device=model.device)
        for test_data, test_target in test_dl:
            metrics = task.metrics if context == "eval" else None
            test_loss += _eval_batch(
                model=model,
                data=test_data,
                target=test_target,
                loss_fn=task.loss_fn,
                metrics=metrics,
            )
        return test_loss


def _eval_batch(
    model: Model,
    data: DataItem,
    target: torch.Tensor,
    loss_fn: nn.Module,
    metrics: Sequence[TaskMetric] | None = None,
) -> torch.Tensor:
    data = data.to(device=model.device)
    target = target.to(device=model.device)
    if Settings().disable_cudnn:
        with torch.backends.cudnn.flags(enabled=False):
            out = model(data)
    else:
        out = model(data)
    loss: torch.Tensor = loss_fn(out, target)
    if metrics is not None:
        with torch.no_grad():
            for metric in metrics:
                metric.update(out.detach().cpu(), target.detach().cpu())
    return loss
