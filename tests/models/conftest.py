from __future__ import annotations

import pytest
import torch


@pytest.fixture(
    params=[
        None,
        {"dates": torch.cat([torch.arange(0, 365, dtype=torch.long).unsqueeze(0)] * 16)},
    ],
    ids=["without dates", "with dates"],
)
def timeseries_meta_data(request) -> dict[str, torch.Tensor]:  # type: ignore
    return request.param  # type: ignore


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
