import itertools

import pytest

from eurocropsmeta.dataset.task import Task
from eurocropsmeta.models.base import ModelBuilder

from ..utils import DenseNNBuilder, interval_tasks

NUM_CLASSES = 3


@pytest.fixture
def tasks() -> list[Task]:
    task_ds = interval_tasks(num_classes=NUM_CLASSES)
    return list(itertools.islice(iter(task_ds), 10))


@pytest.fixture
def model_builder() -> ModelBuilder:
    return DenseNNBuilder(in_size=1, out_size=NUM_CLASSES)
