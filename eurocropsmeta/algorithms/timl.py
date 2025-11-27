import logging
from collections.abc import Iterable
from typing import Literal, cast

import torch
from torch import nn
from torch.optim import Adam, Optimizer

from eurocropsmeta.algorithms.base import MetaLearnAlgorithm, MetaLearnAlgorithmConfig
from eurocropsmeta.algorithms.inner_loop import InnerLoopLearner, SimpleAdam, SimpleSGD
from eurocropsmeta.algorithms.utils import update_module
from eurocropsmeta.models.base import Model, ModelBuilder

logger = logging.getLogger(__name__)


class TIMLConfig(MetaLearnAlgorithmConfig):
    """Config for TIML meta-learning algorithm."""

    __doc__ = (
        cast(str, MetaLearnAlgorithmConfig.__doc__)
        + """
        encoder: Whether to use TIML Task encoder or not.
            If this is False, task information is just appended to the time series.
        inner_lr: Learning rate used in inner loop
        meta_lr: Learn rate to learn inner_lr. If this is 0.0, the inner learning
        rate is kept fixed.
        encoder_lr: Learning rate used for TIML Task encoder.
        first_order: Whether to ignore higher-order derivates in outer update.
        """
    )

    name: Literal["TIML"] = "TIML"

    encoder: bool = True
    inner_lr: float
    meta_lr: float = 0.0
    encoder_lr: float | None = None

    first_order: bool = False


class TIML(MetaLearnAlgorithm[TIMLConfig]):
    """Implementation of original TIML algorithm.

    Args:
        config: Algorithm config.
        model_builder: Builder to use to create initial model.
    """

    def __init__(
        self,
        config: TIMLConfig,
        model_builder: ModelBuilder,
    ):
        super().__init__(config=config, model_builder=model_builder)
        if self.config.encoder:
            self._encoder_optimizer = self.build_encoder_optimizer()
            logger.info("Using TIML algorithm with encoder.")
        else:
            logger.info("Using TIML algorithm without encoder.")

    def build_inner_optimizer(self) -> InnerLoopLearner:
        """Build optimizer for inner loop."""
        match self.config.inner_optimizer:
            case "SGD":
                return SimpleSGD(lr=self.config.inner_lr, meta_lr=self.config.meta_lr)
            case "Adam":
                return SimpleAdam(lr=self.config.inner_lr, meta_lr=self.config.meta_lr)

    def outer_parameters(self) -> list[nn.Parameter]:
        if self.config.reset_head:
            return list(self.model.backbone.parameters())
        return list(self.model.backbone.parameters()) + list(self.model.head.parameters())

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(cast(nn.Module, self.model.encoder).parameters())

    def build_encoder_optimizer(self) -> Optimizer:
        """Build optimizer for encoder within outer loop."""
        model_params = self.encoder_parameters()
        params: list[dict[str, Iterable[nn.Parameter] | float | None]] = [
            {
                "params": model_params,
                "lr": self.config.encoder_lr,
            }
        ]

        return Adam(params)

    def inner_update(self, model: Model, train_loss: torch.Tensor, inner_step: int) -> None:
        updates = self.inner_learner.update(
            train_loss,
            parameters=list(model.named_parameters()),
            create_graph=not self.config.first_order,
            inner_step=inner_step,
        )
        update_module(model, updates=updates)
