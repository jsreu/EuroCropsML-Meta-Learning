import logging
from typing import TypeVar

import torch
import torch.nn as nn

ModuleT = TypeVar("ModuleT")

logger = logging.getLogger(__name__)


def clone_module(
    module: ModuleT, memo: dict[int, nn.Parameter | torch.Tensor] | None = None
) -> ModuleT:
    """Create a copy of the  module and connect it to the computation graph.

    This function is adapted from learn2learn.
    See https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

    Args:
        module: Module to be cloned.
        memo: Parameter memoization to avoid duplicate clones. Only used internally.

    Returns:
        The cloned module.
    """
    if not isinstance(module, nn.Module):
        return module

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, "_parameters"):
        for param_key in module._parameters:
            if (param := module._parameters[param_key]) is not None:
                param_ptr = param.data_ptr()
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]  # type: ignore[assignment]
                else:
                    cloned_param: nn.Parameter = param.clone()  # type: ignore[assignment]
                    clone._parameters[param_key] = cloned_param
                    memo[param_ptr] = cloned_param

    # Third, handle the buffers if necessary
    if hasattr(clone, "_buffers"):
        for buffer_key in module._buffers:
            if (buff := module._buffers[buffer_key]) is not None:
                if not buff.requires_grad:
                    continue
                buff_ptr = buff.data_ptr()
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned_buffer = buff.clone()
                    clone._buffers[buffer_key] = cloned_buffer
                    memo[buff_ptr] = cloned_buffer

    # Then, recurse for each submodule
    if hasattr(clone, "_modules"):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, "flatten_parameters"):
        clone = clone._apply(lambda x: x)
    return clone


def update_module(
    module: ModuleT,
    updates: list[torch.Tensor | None] | None = None,
    memo: dict[nn.Parameter | torch.Tensor, nn.Parameter | torch.Tensor] | None = None,
) -> ModuleT:
    """Update the parameters of a module in-place, in a way that preserves differentiability.

    This function is adapted from learn2learn.
    See https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

    Args:
        module: The module to update.
        updates: A list of gradients for each parameter of the model.
            If None, will use the tensors in .update attributes.
        memo: Parameter memoization to avoid duplicate clones. Only used internally.

    Returns:
        Updated module.
    """
    if not isinstance(module, nn.Module):
        return module
    if memo is None:
        memo = {}

    if updates is not None:
        params = list(module.parameters())
        if len(updates) != len(list(params)):
            logger.warning(
                "Parameters and updates have different length. %s vs %s",
                len(params),
                len(updates),
            )
        for param, g in zip(params, updates):
            param.update = g  # type: ignore[attr-defined]

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is None:
            continue
        if p in memo:
            module._parameters[param_key] = memo[p]  # type: ignore[assignment]
        else:
            if p is not None and hasattr(p, "update"):
                if p.update is None:
                    continue
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, "update"):
                if buff.update is None:
                    continue
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, "flatten_parameters"):
        module._apply(lambda x: x)
    return module
