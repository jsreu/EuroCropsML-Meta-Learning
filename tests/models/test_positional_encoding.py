import math

import numpy as np
import pytest
import torch
from eurocropsml.dataset.base import DataItem

from eurocropsmeta.models.positional_encoding import PositionalEncoding


@pytest.fixture
def positional_encodings_zero_ones() -> tuple[list, list]:
    timesteps = torch.arange(366).unsqueeze(1)
    # Calculate the positional encoding p
    p = torch.zeros(366, 128)
    div_term = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(float(1000)) / 128))
    p[:, 0::2] = torch.sin(timesteps * div_term)
    p[:, 1::2] = torch.cos(timesteps * div_term)

    zeroes = torch.where(p == 0)
    ones = torch.where(p == 1)
    zipped_zeroes = list(zip(zeroes[0].tolist(), zeroes[1].tolist()))
    zipped_ones = list(zip(ones[0].tolist(), ones[1].tolist()))

    return zipped_zeroes, zipped_ones


@pytest.fixture
def encoder() -> PositionalEncoding:
    return PositionalEncoding(d_hid=128, pos_enc_len=366)


@pytest.fixture
def linear_layer() -> torch.nn.Linear:
    return torch.nn.Linear(13, 128)


@pytest.fixture
def test_data_item_padded() -> DataItem:
    data = np.ones((366, 13))
    tensor_data = torch.tensor(data, dtype=torch.float)
    random_days = np.random.choice(np.arange(366), size=100, replace=False)
    sorted_days = np.sort(random_days)
    sorted_days = np.append(sorted_days, [-1] * 10)

    return DataItem(
        data=tensor_data.unsqueeze(0),
        meta_data={"dates": torch.tensor(sorted_days).unsqueeze(0)},
    )


@pytest.fixture
def test_data_item() -> DataItem:
    data = np.ones((100, 13))
    # padding for batching
    data = np.concatenate((data, -1 * np.ones((10, 13))), axis=0) # type: ignore[assignment]
    tensor_data = torch.tensor(data, dtype=torch.float)
    random_days = np.random.choice(np.arange(366), size=100, replace=False)
    sorted_days = np.sort(random_days)
    sorted_days = np.append(sorted_days, [-1] * 10)  # padding for batching

    return DataItem(
        data=tensor_data.unsqueeze(0),
        meta_data={"dates": torch.tensor(sorted_days).unsqueeze(0)},
    )


def test_positional_encoding(
    encoder: PositionalEncoding,
    test_data_item: DataItem,
    linear_layer: torch.nn.Linear,
    positional_encodings_zero_ones: torch.Tensor,
) -> None:

    data = test_data_item.data
    dates = test_data_item.meta_data["dates"]
    # padded values
    valid_dates = dates[dates != -1]
    data = linear_layer(data)
    new_data = encoder(data, dates)

    assert new_data.size() == data.size()
    data = data.squeeze()
    new_data = new_data.squeeze()

    zipped_zeroes = positional_encodings_zero_ones[0]
    zipped_ones = positional_encodings_zero_ones[1]

    for idx, datapoint in enumerate(data):
        new_datapoint = new_data[idx]
        if idx < len(valid_dates):
            for hid_dim_index, dtp in enumerate(datapoint):
                if (idx, hid_dim_index) in zipped_zeroes:
                    assert torch.equal(new_datapoint[hid_dim_index], dtp)
                elif (idx, hid_dim_index) in zipped_ones:
                    assert torch.equal(new_datapoint[hid_dim_index], dtp + 1)
                else:
                    assert not torch.equal(new_datapoint[hid_dim_index], dtp)
        else:
            # for padded values nothing should be added
            assert torch.equal(new_datapoint, datapoint)


def test_positional_encoding_padded(
    encoder: PositionalEncoding,
    test_data_item_padded: DataItem,
    linear_layer: torch.nn.Linear,
    positional_encodings_zero_ones: torch.Tensor,
) -> None:
    data = test_data_item_padded.data
    dates = test_data_item_padded.meta_data["dates"]
    # padded values
    valid_dates = dates[dates != -1]
    data = linear_layer(data)
    new_data = encoder(data, dates)

    assert new_data.size() == data.size()

    data = data.squeeze()
    new_data = new_data.squeeze()

    zipped_zeroes = positional_encodings_zero_ones[0]
    zipped_ones = positional_encodings_zero_ones[1]

    for idx, datapoint in enumerate(data):
        new_datapoint = new_data[idx]
        if idx in valid_dates:
            for hid_dim_index, dtp in enumerate(datapoint):
                if (idx, hid_dim_index) in zipped_zeroes:
                    assert torch.equal(new_datapoint[hid_dim_index], dtp)
                elif (idx, hid_dim_index) in zipped_ones:
                    assert torch.equal(new_datapoint[hid_dim_index], dtp + 1)
                else:
                    assert not torch.equal(new_datapoint, datapoint)
        else:
            assert torch.equal(new_datapoint, datapoint)
