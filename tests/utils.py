import math
from typing import cast

import torch
import torch.nn as nn
from eurocropsml.dataset.base import DataItem, LabelledData, TransformDataset
from torch.utils.data import Dataset

from eurocropsmeta.dataset.task import (
    ClassificationTaskDataset,
    WrapClassificationTaskDataset,
)
from eurocropsmeta.models.base import ModelBuilder


class SineDataset(Dataset[LabelledData]):
    """Toy dataset modeling a sine curve.

    Args:
        phase: Phase of sine curve
        size: Number of data points to construct.
    """

    def __init__(self, phase: float, size: int = 100):
        self.data = math.pi * torch.rand((size, 1))
        self.targets = torch.sin(self.data + phase)

    def __getitem__(self, ix: int) -> LabelledData:
        return LabelledData(DataItem(self.data[ix]), self.targets[ix])

    def __len__(self) -> int:
        return self.data.size(0)


def interval_tasks(
    total_classes: int = 20,
    num_classes: int = 3,
    train_size: int = 5,
    test_size: int = 3,
    offset: float = 0.0,
    sample: bool = False,
) -> ClassificationTaskDataset:
    """Construct classification task set based on interval datasets.

    Args:
        total_classes: Total number of classes in dataset.
        num_classes: Number of classes in each task.
        train_size: Datapoints per class in train set.
        test_size: Datapoints per class in test set.
        offset: Offset to add to interval.
        sample: Whether to sample tasks randomly.

    Returns:
        Toy classification task dataset.
    """

    intervals = [
        (offset + 100 * k / total_classes, offset + 100 * (k + 1) / total_classes)
        for k in range(total_classes)
    ]
    intervals = sorted(intervals)
    datasets = {
        n: TransformDataset(IntervalDataset(interval, label=n, size=train_size + test_size))
        for n, interval in enumerate(intervals)
    }

    return ClassificationTaskDataset(
        datasets,
        num_classes=num_classes,
        train_samples_per_class=train_size,
        test_samples_per_class=test_size,
        sample=sample,
        loss_fn=nn.CrossEntropyLoss(),
        metrics_list=["Acc"],
    )


def wrappedinterval_tasks(
    total_sup_classes: int = 5,
    total_classes: int = 20,
    num_classes: int = 3,
    train_size: int = 5,
    test_size: int = 3,
    offset: float = 0.0,
    sample: bool = False,
) -> WrapClassificationTaskDataset:
    """Construct wrap classification task set based on interval datasets.

    Args:
        total_sup_classes: Total number of superclasses in dataset.
        total_classes: Total number of classes in dataset.
        num_classes: Number of classes in each task.
        train_size: Datapoints per class in train set.
        test_size: Datapoints per class in test set.
        offset: Offset to add to interval.
        sample: Whether to sample tasks randomly.

    Returns:
        Wrapped toy classification task dataset.
    """
    superclasses = list(range(0, total_sup_classes))

    intervals = [
        (offset + 100 * k / total_classes, offset + 100 * (k + 1) / total_classes)
        for k in range(total_classes)
    ]
    intervals = sorted(intervals)
    datasets = {
        n: TransformDataset(IntervalDataset(interval, label=n, size=train_size + test_size))
        for n, interval in enumerate(intervals)
    }

    wrapped_dataset = dict.fromkeys(superclasses, datasets)

    return WrapClassificationTaskDataset(
        wrapped_dataset,
        num_classes=num_classes,
        train_samples_per_class=train_size,
        test_samples_per_class=test_size,
        sample=sample,
        loss_fn=nn.CrossEntropyLoss(),
        metrics_list=["Acc"],
    )


class IntervalDataset(Dataset[LabelledData]):
    """Toy classification dataset containing points in given interval.

    Args:
        interval: Real interval points are sampled from.
        label: Constant label to give targets.
        size: Number of data points to construct.
    """

    def __init__(self, interval: tuple[float, float], label: int, size: int):
        start, end = interval
        self.data = start + (end - start) * torch.rand((size, 1))
        self.targets = label * torch.ones(size, dtype=torch.long)

    def __getitem__(self, ix: int) -> LabelledData:
        return LabelledData(DataItem(self.data[ix]), self.targets[ix])

    def __len__(self) -> int:
        return self.data.size(0)


class DataItemToTensor(nn.Module):
    """Module for getting a specified attribute from a DataItem.

    Args:
        model_input_att: Name of attribute to get from DataItem
        metadata: True if attribute is a metadata attribute
    """

    def __init__(self, model_input_att: str, metadata: bool = False):
        super().__init__()
        self.model_input_att = model_input_att
        self.metadata = metadata

    def forward(self, ipt: DataItem) -> torch.Tensor:
        """Forward pass."""
        if self.metadata:
            ipt = getattr(ipt, self.model_input_att)
        return cast(torch.Tensor, getattr(ipt, self.model_input_att))


class DenseNNBuilder(ModelBuilder):
    """Builder for two-layer dense neural network.

    Args:
        hidden_size: Size of hidden layer
        out_size: Size of output layer
    """

    def __init__(self, in_size: int, hidden_size: int = 32, out_size: int = 1):
        self._in_size = in_size
        self._hidden_size = hidden_size
        self._out_size = out_size

    def build_backbone(self) -> nn.Sequential:
        """Backbone to pre-train."""
        return nn.Sequential(
            DataItemToTensor("data"),
            nn.Linear(self._in_size, self._hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(inplace=True),
        )

    def build_classification_head(self, num_classes: int) -> nn.Linear:
        """Prediction head."""
        return nn.Linear(self._hidden_size, num_classes)
