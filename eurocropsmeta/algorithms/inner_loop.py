import math
from abc import abstractmethod
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.autograd import grad


class InnerLoopLearner(nn.Module):
    """Inner loop learner for meta-learning."""

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor,
        parameters: list[tuple[str, nn.Parameter]],
        create_graph: bool,
        inner_step: int,
    ) -> list[torch.Tensor | None]:
        pass

    @abstractmethod
    def optimizer_params(self) -> dict[str, Iterable[nn.Parameter] | float | None]:
        """Parameter config for outer optimizer."""

    def reset_optimizer_state(
        self, reset_values: dict[str, list[torch.Tensor] | None] | None
    ) -> None:
        """Resets the inner learners internal state.

        Some inner loop learners keep an internal state
        that needs to be reset at the start of each new
        task adaption. Optionally overwrite this reset
        method in such cases. By default, the reset method
        will do nothing.
        """
        pass


class SimpleSGD(InnerLoopLearner):
    """A simplified version of torch.optim.SGD to be used for the inner loop."""

    def __init__(self, lr: float, meta_lr: float = 0.0):
        super().__init__()

        self.meta_lr = meta_lr
        self.learn_lr = meta_lr != 0.0
        self.lr = nn.Parameter(torch.tensor(lr), requires_grad=self.learn_lr)

    def update(
        self,
        loss: torch.Tensor,
        parameters: list[tuple[str, nn.Parameter]],
        create_graph: bool,
        inner_step: int,
    ) -> list[torch.Tensor | None]:
        params = [p for _, p in parameters]
        grads: tuple[torch.Tensor | None, ...] = grad(
            loss, params, create_graph=create_graph, allow_unused=True
        )
        updates = [-self.lr * g if g is not None else None for g in grads]
        return updates

    def optimizer_params(self) -> dict[str, Iterable[nn.Parameter] | float | None]:
        if self.learn_lr:
            return {"params": self.parameters(), "lr": self.meta_lr}
        return {}


class SimpleAdam(InnerLoopLearner):
    """A simplified version of torch.optim.Adam to be used for the inner loop."""

    def __init__(
        self,
        lr: float,
        meta_lr: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        super().__init__()

        self.meta_lr = meta_lr
        self.learn_lr = meta_lr != 0.0
        self.lr = nn.Parameter(torch.tensor(lr), requires_grad=self.learn_lr)

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.exp_avgs: list[torch.Tensor | None] | None = None
        self.exp_avg_sqs: list[torch.Tensor | None] | None = None

    def reset_optimizer_state(
        self,
        reset_values: dict[str, list[torch.Tensor] | None] | None,
    ) -> None:
        """Reset the exponentially moving average momentum buffers."""
        # parse provided reset values if applicable
        if reset_values is not None and "exp_avgs" in reset_values:
            reset_exp_avgs = reset_values["exp_avgs"]
        else:
            reset_exp_avgs = None
        if reset_values is not None and "exp_avg_sqs" in reset_values:
            reset_exp_avg_sqs = reset_values["exp_avg_sqs"]
        else:
            reset_exp_avg_sqs = None
        # reset first order momentum buffers
        if reset_exp_avgs is not None and self.exp_avgs is not None:
            for i, p in enumerate(self.exp_avgs):
                p_init = reset_exp_avgs[i] if i < len(reset_exp_avgs) else None
                if (
                    p is not None and p_init is not None and p.shape == p_init.shape
                ):  # could match buffer -> use as init value
                    self.exp_avgs[i] = p_init
                elif p is not None:  # could not match buffer -> init as zero
                    self.exp_avgs[i] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                else:  # receives no grad info -> init as None
                    self.exp_avgs[i] = None
        else:
            self.exp_avgs = None
        # reset second order momentum buffers
        if reset_exp_avg_sqs is not None and self.exp_avg_sqs is not None:
            for i, p in enumerate(self.exp_avg_sqs):
                p_init = reset_exp_avg_sqs[i] if i < len(reset_exp_avg_sqs) else None
                if (
                    p is not None and p_init is not None and p.shape == p_init.shape
                ):  # could match buffer -> use as init value
                    self.exp_avg_sqs[i] = p_init
                elif p is not None:  # could not match buffer -> init as zero
                    self.exp_avg_sqs[i] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                else:  # receives no grad info -> init as None
                    self.exp_avg_sqs[i] = None
        else:
            self.exp_avg_sqs = None

    def update(
        self,
        loss: torch.Tensor,
        parameters: list[tuple[str, nn.Parameter]],
        create_graph: bool,
        inner_step: int,
    ) -> list[torch.Tensor | None]:
        params = [p for _, p in parameters]
        grads: tuple[torch.Tensor | None, ...] = grad(loss, params, create_graph=create_graph)

        # lazily init exponentially moving average buffers if necessary
        if self.exp_avgs is None:
            self.exp_avgs = [
                (
                    torch.zeros_like(g, memory_format=torch.preserve_format)
                    if g is not None
                    else None
                )
                for g in grads
            ]
        if self.exp_avg_sqs is None:
            self.exp_avg_sqs = [
                (
                    torch.zeros_like(g, memory_format=torch.preserve_format)
                    if g is not None
                    else None
                )
                for g in grads
            ]

        updates: list[torch.Tensor | None] = []
        for i, g in enumerate(grads):
            if g is None:
                update = None
            else:
                exp_avg = cast(torch.Tensor, cast(list[torch.Tensor | None], self.exp_avgs)[i])
                exp_avg_sq = cast(
                    torch.Tensor, cast(list[torch.Tensor | None], self.exp_avg_sqs)[i]
                )

                # decay the first and second moment running average coefficient
                exp_avg.lerp_(g, 1 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(g, g.conj(), value=1 - self.beta2)

                # bias correction coefficients
                bias_correction1 = 1 - self.beta1 ** (inner_step + 1)
                bias_correction2 = 1 - self.beta2 ** (inner_step + 1)
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # momentum scaled update
                step_size = self.lr / bias_correction1
                step_size_neg = step_size.neg()
                # Unlike torch.optim.Adam we add eps**2 inside of the sqrt() instead of adding
                # eps outside. This was necessary for being able to differentiate through Adam.
                denom = (exp_avg_sq + self.eps**2).sqrt() / (bias_correction2_sqrt * step_size_neg)
                update = exp_avg.clone() / denom

            updates.append(update)
        return updates

    def optimizer_params(self) -> dict[str, Iterable[nn.Parameter] | float | None]:
        if self.learn_lr:
            return {"params": self.parameters(), "lr": self.meta_lr}
        return {}
