from typing import Literal, cast

import torch
from torch import nn

from eurocropsmeta.algorithms.base import MetaLearnAlgorithm, MetaLearnAlgorithmConfig
from eurocropsmeta.algorithms.inner_loop import InnerLoopLearner, SimpleAdam, SimpleSGD
from eurocropsmeta.algorithms.utils import update_module
from eurocropsmeta.models.base import Model, ModelBuilder


class ANILConfig(MetaLearnAlgorithmConfig):
    """Config for ANIL meta-learning algorithm."""

    __doc__ = (
        cast(str, MetaLearnAlgorithmConfig.__doc__)
        + """
        inner_lr: Learning rate used in inner loop
        meta_lr: Learn rate to learn inner_lr. If this is 0.0, the inner learning
        rate is kept fixed.
        """
    )

    name: Literal["ANIL"] = "ANIL"

    inner_lr: float
    meta_lr: float = 0.0


class ANIL(MetaLearnAlgorithm[ANILConfig]):
    """Implementation of ANIL algorithm.

    Args:
        config: Algorithm config.
        model_builder: Builder to use to create initial model.
    """

    def __init__(self, config: ANILConfig, model_builder: ModelBuilder):
        super().__init__(config=config, model_builder=model_builder)

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
        return list(self.model.parameters())

    def inner_update(self, model: Model, train_loss: torch.Tensor, inner_step: int) -> None:
        updates = self.inner_learner.update(
            train_loss,
            parameters=list(model.head.named_parameters()),
            create_graph=True,
            inner_step=inner_step,
        )
        update_module(model.head, updates=updates)
