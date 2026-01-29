import logging
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch.autograd import grad

from eurocropsmeta.algorithms.base import MetaLearnAlgorithm, MetaLearnAlgorithmConfig
from eurocropsmeta.algorithms.inner_loop import InnerLoopLearner
from eurocropsmeta.algorithms.utils import update_module
from eurocropsmeta.models.base import Model, ModelBuilder

logger = logging.getLogger(__name__)


class ALFAConfig(MetaLearnAlgorithmConfig):
    """Config for ALFA meta-learning algorithm."""

    __doc__ = (
        cast(str, MetaLearnAlgorithmConfig.__doc__)
        + """
        meta_lr: Learning rate used in inner learner. If this is None, re-use outer_lr.
        beta_min: Minimum value of learned beta parameter (1-weight_decay).

        first_order: Whether to ignore higher-order derivates in outer update.
        layer_depth: Depth of module hierarchy to use as cut off for layer splitting.
        """
    )

    name: Literal["ALFA"] = "ALFA"

    meta_lr: float | None = None
    beta_min: float = 0.9

    first_order: bool = False
    layer_depth: int = 2

    @field_validator("meta_lr")
    @classmethod
    def meta_lr_default(cls, v: float | None, info: ValidationInfo) -> float:
        """Set meta lr to outer lr if not otherwise given."""
        if v is None:
            outer_lr: float = info.data["outer_lr"]
            return outer_lr
        return v


class ALFA(MetaLearnAlgorithm[ALFAConfig]):
    """Implementation of original ALFA algorithm.

    See
    - Baik, S., Choi, M., Choi, J., Kim, H., & Lee, K. M. (2020). Meta-Learning With
    Adaptive Hyperparameters. CoRR, (), .

    Note:
    Contrary to the original implementation, we do not learn separate learning rates
    for each inner step, but only for each model layer.

    Args:
        config: Algorithm config.
        model_builder: Builder to use to create initial model.
    """

    def __init__(
        self,
        config: ALFAConfig,
        model_builder: ModelBuilder,
    ):
        super().__init__(config=config, model_builder=model_builder)

    def build_inner_optimizer(self) -> InnerLoopLearner:
        """Build optimizer for inner loop."""
        logger.info(
            f"{self.config.inner_optimizer} was specified as optimizer "
            "for the inner loop but ignored as ALFA uses a custom inner "
            "loop optimizer."
        )
        num_layers = ALFALearner.compute_num_layers(
            self.model_builder.build_classification_model(1, torch.device("cpu")),
            self.config.layer_depth,
        )
        return ALFALearner(
            meta_lr=self.config.meta_lr,
            num_layers=num_layers,
            layer_depth=self.config.layer_depth,
            beta_min=self.config.beta_min,
        )

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


class ALFALearner(InnerLoopLearner):
    """Module for learning hyper-parameters from model params and grads.

    Args:
        meta_lr: Learning rate to use to adapt module during outer loop.
        beta_min: Minimum value of learned beta parameter (1-weight_decay).
        num_layers: Number of layers of model to adapt.
        num_blocks: Number of MLP blocks of learner module
        layer_depth: Depth of module hierarchy to use as cut off for layer splitting.
    """

    def __init__(
        self,
        meta_lr: float | None,
        beta_min: float,
        num_layers: int,
        num_blocks: int = 3,
        layer_depth: int = 2,
    ):
        super().__init__()

        self.meta_lr = meta_lr
        self.beta_min = beta_min
        self.num_layers = num_layers
        self.layer_depth = layer_depth
        self._hidden_channels = 2 * num_layers
        self.sequential = nn.Sequential(*[self._mlp() for _ in list(range(num_blocks))])
        self.alpha_multipliers = nn.Parameter(self.meta_lr * torch.ones(num_layers))
        self.recorded_params: list[dict[str, dict[str, float]]] | None = None

    @classmethod
    def compute_num_layers(cls, model: Model, layer_depth: int) -> int:
        return len({_reduce_param_name(name, layer_depth) for name, _ in model.named_parameters()})

    def _mlp(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self._hidden_channels,
                self._hidden_channels,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        layer_means: torch.Tensor,
        layer_grad_means: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([layer_means, layer_grad_means])
        out = self.sequential(x)

        alphas = out[: self.num_layers]
        # TODO: This can lead to negative learning rates. How do we avoid this?
        alphas = self.alpha_multipliers * (2 * alphas.sigmoid())

        betas = out[self.num_layers :]
        betas = (1 - self.beta_min) * betas.sigmoid() + self.beta_min * torch.ones(
            self.num_layers,
            device=out.device,
        )

        return alphas, betas

    @property
    def record_params(self) -> bool:
        return self.recorded_params is not None

    @record_params.setter
    def record_params(self, value: bool) -> None:
        if not value:
            self.recorded_params = None
        if value and self.recorded_params is None:
            self.recorded_params = []

    def update(
        self,
        loss: torch.Tensor,
        parameters: list[tuple[str, nn.Parameter]],
        create_graph: bool,
        inner_step: int,
    ) -> list[torch.Tensor | None]:
        if (inner_step == 0) and self.recorded_params is not None:
            self.recorded_params = []

        layer_mapping = _compute_layer_mapping(parameters, self.layer_depth)
        assert len(layer_mapping) == self.num_layers
        params = [p for _, p in parameters]
        grads: tuple[torch.Tensor | None, ...] = grad(loss, params, create_graph=create_graph)
        params = [p for _, p in parameters]

        layer_means = torch.stack(
            [
                _layer_mean([params[ix] for ix in layer_mapping[layer_name]])
                for layer_name in layer_mapping
            ]
        )
        layer_grad_means = torch.stack(
            [
                _layer_mean(
                    [
                        cast(torch.Tensor, grads[ix])
                        for ix in layer_mapping[layer_name]
                        if grads[ix] is not None
                    ]
                )
                for layer_name in layer_mapping
            ]
        )

        alphas, betas = self(layer_means, layer_grad_means)

        if self.recorded_params is not None:
            recorded_params = {
                layer_name: {"alpha": alphas[n].item(), "beta": betas[n].item()}
                for n, layer_name in enumerate(layer_mapping)
            }
            self.recorded_params.append(recorded_params)

        updates: list[torch.Tensor | None] = []
        for n, layer_name in enumerate(layer_mapping):
            for ix in layer_mapping[layer_name]:
                g = grads[ix]
                p = params[ix]
                update = None if g is None else (betas[n] - 1) * p - alphas[n] * g
                updates.append(update)
        return updates

    def optimizer_params(
        self,
    ) -> dict[str, Iterable[nn.Parameter] | float | None]:
        return {"params": self.parameters(), "lr": self.meta_lr}


def _compute_layer_mapping(
    parameters: Sequence[tuple[str, nn.Parameter]], layer_depth: int
) -> OrderedDict[str, list[int]]:
    param_groups = defaultdict(list)
    for ix, (name, _) in enumerate(parameters):
        param_groups[_reduce_param_name(name, layer_depth)].append(ix)
    return OrderedDict(param_groups)


def _layer_mean(layer_params: list[torch.Tensor] | list[nn.Parameter]) -> torch.Tensor:
    if not layer_params:
        return torch.tensor(0.0)
    total_size = sum(sum(p.size()) for p in layer_params)
    layer_sum = torch.stack([p.sum().detach() for p in layer_params]).sum()
    return layer_sum / total_size


def _reduce_param_name(name: str, layer_depth: int) -> str:
    return ".".join(name.split(".")[:layer_depth])
