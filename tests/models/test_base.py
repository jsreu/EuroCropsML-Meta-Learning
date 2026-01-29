from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from eurocropsmeta.models.base import Model, ModelBuilder

from ..utils import DenseNNBuilder

NUM_CLASSES = 10


@pytest.fixture
def model_builder() -> ModelBuilder:
    return DenseNNBuilder(in_size=10)


@pytest.fixture
def classification_model(model_builder: ModelBuilder) -> Model:
    return model_builder.build_classification_model(
        num_classes=NUM_CLASSES, device=torch.device("cpu")
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path.joinpath("model_checkpoint")
    checkpoint.mkdir()
    return checkpoint


@pytest.mark.parametrize("model", ["classification_model"])
@pytest.mark.parametrize("load_head", [True, False])
def test_save_load_model(model: Model, checkpoint: Path, load_head: bool, request: Any) -> None:
    model = request.getfixturevalue(model)
    model.save(checkpoint)

    loaded_model = deepcopy(model)
    loaded_model.load(checkpoint=checkpoint, load_head=load_head)

    if load_head:
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert (p1 == p2).all()
    else:
        for p1, p2 in zip(model.backbone.parameters(), loaded_model.backbone.parameters()):
            assert (p1 == p2).all()
