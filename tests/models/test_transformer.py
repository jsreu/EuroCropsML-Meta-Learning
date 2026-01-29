import numpy as np
import pytest
import torch
from eurocropsml.dataset.base import DataItem

from eurocropsmeta.models.transformer import TransformerConfig, TransformerModelBuilder


@pytest.fixture
def config() -> TransformerConfig:
    return TransformerConfig(
        n_heads=4,
        in_channels=12,
        d_model=128,
        dim_fc=2048,
        num_layers=1,
        pos_enc_len=365,
    )


@pytest.fixture
def test_data_item() -> DataItem:
    data = np.ones((100, 12))
    tensor_data = torch.tensor(data, dtype=torch.float)
    random_days = np.random.choice(np.arange(366), size=100, replace=False)
    sorted_days = np.sort(random_days)
    return DataItem(
        data=tensor_data.unsqueeze(0),
        meta_data={"dates": torch.tensor(sorted_days).unsqueeze(0)},
    )


@pytest.fixture
def model_builder(config: TransformerConfig) -> TransformerModelBuilder:
    return TransformerModelBuilder(config=config)


def test_classification_forward(
    model_builder: TransformerModelBuilder,
    test_data_item: DataItem,
    device: torch.device,
) -> None:
    num_classes = 10
    model = model_builder.build_classification_model(num_classes, device)

    out = model(test_data_item)

    assert out.size() == (test_data_item.data.size(0), num_classes)


def test_reset_head(
    model_builder: TransformerModelBuilder,
    test_data_item: DataItem,
    device: torch.device,
) -> None:
    num_classes = 10
    model = model_builder.build_classification_model(num_classes, device)

    out = model(test_data_item)
    assert out.size() == (test_data_item.data.size(0), num_classes)

    num_new_classes = 7
    model_builder.reset_head(model, num_new_classes)
    out = model(test_data_item)

    assert out.size() == (test_data_item.data.size(0), num_new_classes)
