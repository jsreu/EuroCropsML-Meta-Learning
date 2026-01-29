import torch

from eurocropsmeta.dataset.utils import remap_values


def test_remap_values() -> None:
    value_map = {2: 0, 1: 1, 0: 0}
    data = torch.tensor([1, 1, 2, 2, 0, 0])
    result = remap_values(value_map, data)
    expected = torch.tensor([1, 1, 0, 0, 0, 0])

    assert (result == expected).all()
