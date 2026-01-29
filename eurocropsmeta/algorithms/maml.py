import logging
from typing import Literal, cast

import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import nn

from eurocropsmeta.algorithms.base import MetaLearnAlgorithm, MetaLearnAlgorithmConfig
from eurocropsmeta.algorithms.inner_loop import InnerLoopLearner, SimpleAdam, SimpleSGD
from eurocropsmeta.algorithms.utils import update_module
from eurocropsmeta.models.base import Model, ModelBuilder

logger = logging.getLogger(__name__)


class MAMLConfig(MetaLearnAlgorithmConfig):
    """Config for MAML meta-learning algorithm."""

    __doc__ = (
        cast(str, MetaLearnAlgorithmConfig.__doc__)
        + """
        inner_lr: Learning rate used in inner loop
        meta_lr: Learn rate to learn inner_lr. If this is 0.0, the inner learning
            rate is kept fixed.
        first_order: Whether to ignore higher-order derivates in outer update.
        """
    )

    name: Literal["MAML"] = "MAML"

    inner_lr: float
    meta_lr: float = 0.0

    first_order: bool = False

    @field_validator("init_inner_momentum_from_outer")
    @classmethod
    def init_momentum_only_for_maml(cls, v: bool, info: ValidationInfo) -> bool:
        inner_optimizer: str = info.data["inner_optimizer"]
        outer_optimizer: str = info.data["outer_optimizer"]
        if v and not ("Adam" in inner_optimizer and "Adam" in outer_optimizer):
            logger.warning(
                "Initializing the exponentially running average momentum buffers "
                "of the inner loop optimizer from the buffers of the outer loop "
                "optimizer is only possible for the MAML algorithm and if "
                "Adam is used for both loops. This is not the case for the "
                "current configuration. Therefore, the chosen buffer initialization "
                "strategy will be ignored."
            )
            return False
        else:
            return v


class MAML(MetaLearnAlgorithm[MAMLConfig]):
    """Implementation of original MAML algorithm.

    Args:
        config: Algorithm config.
        model_builder: Builder to use to create initial model.
    """

    def __init__(
        self,
        config: MAMLConfig,
        model_builder: ModelBuilder,
    ):
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
            parameters=list(model.named_parameters()),
            create_graph=not self.config.first_order,
            inner_step=inner_step,
        )
        update_module(model, updates=updates)
