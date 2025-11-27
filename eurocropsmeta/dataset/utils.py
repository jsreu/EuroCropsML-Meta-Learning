import logging

import torch

logger = logging.getLogger(__name__)


def remap_values(value_map: dict[int, int], t: torch.Tensor) -> torch.Tensor:
    """Change integer values of tensor according to value_map."""
    for source, target in value_map.items():
        t = t.masked_fill(t == source, target)
    return t
