from __future__ import annotations

import itertools
import logging
import math
import os
import random
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Generic, Literal, TypeVar

import numpy as np
import torch.nn as nn
from eurocropsml.dataset.base import TensorDataset, TransformDataset
from torch.utils.data import DataLoader, Dataset

from eurocropsmeta.dataset.utils import remap_values
from eurocropsmeta.train.utils import TaskMetric, get_metrics

logger = logging.getLogger(__name__)


TaskIx = TypeVar("TaskIx")


@dataclass
class Task:
    """Class specifying a training task.

    Args:
        task_id: Task identifier.
        train_set: Task train set.
        loss_fn: Loss function appropriate for task.
        metrics: Additional metrics evaluated during testing/validation.
        num_classes: Number of classes per task
        test_set: Optional task test set.
        val_set: Optional task validation set.
    """

    task_id: str
    train_set: TransformDataset
    loss_fn: nn.Module
    metrics: Sequence[TaskMetric]
    num_classes: int
    test_set: TransformDataset | None = None
    val_set: TransformDataset | None = None

    DATALOADER_NUM_WORKERS: ClassVar[int | None] = 0

    def _build_dl(
        self,
        ds: TransformDataset,
        batch_size: int,
        mode: Literal["train", "test"] = "train",
    ) -> DataLoader:
        num_workers = self.DATALOADER_NUM_WORKERS
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            drop_last=False,
            collate_fn=ds.collate_fn,
            persistent_workers=bool(num_workers),
        )

    def train_dl(self, batch_size: int | None = None, num_batches: int | None = None) -> DataLoader:
        """Build dataloader serving train data with given batch size or number of batches."""
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.train_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.train_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.train_set, batch_size, mode="train")

    def val_dl(self, batch_size: int | None = None, num_batches: int | None = None) -> DataLoader:
        """Build dataloader serving validation data with given batch size or number of batches.

        Raises ValueError if val_set is None.
        """
        if self.val_set is None:
            raise ValueError(f"No validation set given for task {self.task_id}.")
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.val_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.val_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.val_set, batch_size, mode="test")

    def test_dl(self, batch_size: int | None = None, num_batches: int | None = None) -> DataLoader:
        """Build dataloader serving test data with given batch size or number of batches.

        Raises ValueError if test_set is None.
        """
        if self.test_set is None:
            raise ValueError(f"No test set given for task {self.task_id}.")
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.test_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.test_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.test_set, batch_size, mode="test")


class TaskDataset(Dataset, Generic[TaskIx]):
    """Abstract class modeling a dataset of tasks."""

    @abstractmethod
    def __getitem__(self, ix: TaskIx) -> Task | None:
        """Build task for given index.

        Args:
            ix: Task index.

        Returns:
            Task for given index.
            Returns None if no task for this index could be build.
        """

    @abstractmethod
    def task_iter(self) -> Iterator[TaskIx]:
        """Iterator yielding task indices."""

    def __iter__(self) -> Iterator[Task]:
        for task_ix in self.task_iter():
            task = self[task_ix]
            if task is not None:
                yield task

    @abstractmethod
    def num_tasks(self) -> int | None:
        """Total number of tasks of dataset. Returns None if the number of tasks is infinite."""

    def dataloader(
        self,
        batch_size: int,
        prefetch_task_data: bool | int = False,
        num_workers: int | None = 0,
    ) -> DataLoader[Task]:
        """Construct dataloader yielding task batches.

        Args:
            batch_size: Number of tasks returned in each batch.
            prefetch_task_data: Whether to load fetch task data when sampling a new task.
                If this is an integer, fetch that many data points for each task
            num_workers: Number of workers to use for parallel processing.
                If None, use the number of available cpu cores.

        Returns:
            Dataloader constructing and serving tasks.
        """

        if prefetch_task_data:
            task_size: int | None = None
            if isinstance(prefetch_task_data, int):
                task_size = prefetch_task_data
            task_ds: TaskDataset = PrecomputedTaskDataset(self, task_size=task_size)
        else:
            task_ds = self
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        sampler = self.task_iter()

        return DataLoader(
            dataset=task_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=_task_collate,
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
        )


def _task_collate(tasks: list[Task | None]) -> list[Task]:
    return [task for task in tasks if task is not None]


class WrapClassificationTaskDataset(TaskDataset[tuple[int, tuple[int, ...]]]):
    """Class for modeling n-way k-shot classification tasks with multiple super classes.

    First a super class is sampled from which then n-way k-shot classification tasks are sampled.

    Args:
        datasets: Mapping of super class id to corresponding label ids and datasets.
        num_classes: Number of classes per task.
        train_samples_per_class: Number of samples used for training (support) set (per class).
            If None, use all remaining data.
        test_samples_per_class: Number of samples used for test (query) set (per class).
        sample: If True, the dataloader samples a new class combination for each task.
            Otherwise, sequentially iterate through all class combinations.
        loss_fn: Loss function used to calculate the model's loss.
        metrics_list: Additional metrics evaluated during testing/validation.
    """

    def __init__(
        self,
        datasets: Mapping[int, Mapping[int, TransformDataset]],
        num_classes: int,
        train_samples_per_class: int | None,
        test_samples_per_class: int,
        sample: bool,
        loss_fn: nn.Module,
        metrics_list: Sequence[str],
    ):
        self._datasets = datasets

        self._superclasses = list(datasets.keys())
        self.num_total_superclasses = len(datasets.keys())
        if set(self._superclasses) != set(range(self.num_total_superclasses)):
            raise ValueError("Wrapper dataset ids must be numbered consecutively starting at 0.")
        for _, class_dict in datasets.items():
            total_classes = len(class_dict.keys())
            if set(class_dict.keys()) != set(range(total_classes)):
                raise ValueError("Class dataset ids must be numbered consecutively starting at 0.")

        self.sample = sample

        self._num_classes = num_classes
        self._train_samples_per_class = train_samples_per_class
        self._test_samples_per_class = test_samples_per_class

        self.loss_fn = loss_fn
        self.metrics_list = metrics_list

    def __getitem__(self, ix: tuple[int, tuple[int, ...]]) -> Task | None:
        return self._build_task(ix)

    def _build_task(
        self,
        ix: tuple[int, tuple[int, ...]] | str,
        class_splits: dict[int, tuple[list[int], list[int]]] | None = None,
    ) -> Task | None:
        if isinstance(ix, str):
            raw_super_ix, raw_classes = ix.split("-")
            super_ix = int(raw_super_ix)
            classes = tuple([int(c) for c in raw_classes.split("|")])
        else:
            super_ix, classes = ix
        super_dataset = self._datasets[super_ix]

        if min(classes) < 0 or max(classes) >= len(super_dataset.keys()):
            logger.warning(f"Unknown class id in {ix}")
            return None
        if len(classes) != self._num_classes or len(classes) != len(set(classes)):
            logger.warning(f"Invalid class tuple: {classes}")
            return None

        for c in classes:
            if c < 0 or c >= len(super_dataset.keys()):
                raise KeyError(f"Unrecognized class: {c}")
        if class_splits is None:
            class_splits = {}

        train_datasets = []
        test_datasets = []
        for c in classes:
            dataset = super_dataset[c]

            ixs = set(range(len(dataset)))

            if c in class_splits:
                train_ixs, test_ixs = class_splits[c]
            else:
                try:
                    train_ixs, test_ixs = split_data(
                        ixs,
                        dataset,
                        train_samples=self._train_samples_per_class,
                        test_samples=self._test_samples_per_class,
                    )
                except ValueError as e:
                    logger.debug(str(e))
                    continue
                class_splits[c] = (train_ixs, test_ixs)
            test_data = TransformDataset.subset(dataset, test_ixs)
            test_datasets.append(test_data)
            train_data = dataset.subset(train_ixs)
            train_datasets.append(train_data)

        # if no valid task datasets could be built return nothing
        if not train_datasets or not test_datasets:
            return None
        # otherwise concatenate them
        train_set = TransformDataset.concat(train_datasets)
        test_set = TransformDataset.concat(test_datasets)

        class_map = {c: n for n, c in enumerate(classes)}
        class_map_fn = partial(remap_values, class_map)

        train_set.target_transforms += [class_map_fn]
        test_set.target_transforms += [class_map_fn]

        task_id = str(super_ix) + "-" + "|".join(str(c) for c in classes)

        task = Task(
            task_id=task_id,
            train_set=train_set,
            test_set=test_set,
            loss_fn=self.loss_fn,
            metrics=get_metrics(self.metrics_list, num_classes=self._num_classes),
            num_classes=self._num_classes,
        )

        return task

    def task_iter(
        self,
    ) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Iterator yielding task indices."""
        if self.sample:
            sample_counts = [
                sum([len(dataset) for dataset in self._datasets[superclass].values()])
                for superclass in self._superclasses
            ]
            total_count = sum(sample_counts)
            relative_counts = [count / total_count for count in sample_counts]
            while True:
                superclass = np.random.choice(self._superclasses, p=relative_counts)
                classes = list(self._datasets[superclass].keys())
                # check for regions in which fewer classes than num_classes exist
                if len(classes) < self._num_classes:
                    logger.warning(
                        f"Not enough classes for superclass {superclass}. "
                        f"Skipped. (classes={classes})"
                    )
                    continue
                c = random.sample(classes, k=self._num_classes)
                yield superclass, tuple(sorted(c))

        else:
            for sc in self._superclasses:
                classes = list(self._datasets[sc].keys())
                for combination in itertools.combinations(classes, r=self._num_classes):
                    yield sc, combination

    def num_tasks(self) -> int | None:
        """Total number of tasks of dataset. Returns None if the number of tasks is infinite."""
        if self.sample:
            return None
        num_tasks = 0
        for i in range(self.num_total_superclasses):
            classes = len(self._datasets[i].keys())
            num_tasks += math.comb(classes, self._num_classes)
        return num_tasks


class ClassificationTaskDataset(WrapClassificationTaskDataset):
    """Class for modeling n-way k-shot classification tasks.

    Args:
        datasets: Mapping of label id to corresponding dataset.
        num_classes: Number of classes per task.
        train_samples_per_class: Number of samples used for the support set (per class).
            If None, use all remaining data.
        test_samples_per_class: Number of samples used for the test set (per class).
        sample: If True, the dataloader samples a new class combination for each task.
            Otherwise, sequentially iterate through all class combinations.
        loss_fn: Loss function used to calculate the model's loss.
        metrics_list: Additional metrics evaluated during testing/validation.
    """

    def __init__(
        self,
        datasets: Mapping[int, TransformDataset],
        num_classes: int,
        train_samples_per_class: int | None,
        test_samples_per_class: int,
        sample: bool,
        loss_fn: nn.Module,
        metrics_list: Sequence[str],
    ):
        super().__init__(
            datasets={0: datasets},
            num_classes=num_classes,
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
            sample=sample,
            loss_fn=loss_fn,
            metrics_list=metrics_list,
        )


class FixedTaskDataset(TaskDataset[int]):
    """Task dataset for freezing tasks."""

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def __getitem__(self, ix: int) -> Task | None:
        """Build task for given index.

        Args:
            ix: Task index.

        Returns:
            Task for given index.

        Raises:
            KeyError: If no task with the given index can be constructed.
        """
        if ix >= len(self.tasks):
            logger.warning(f"Task Index out of bounds: {ix}>={len(self.tasks)}")
            return None
        return self.tasks[ix]

    def task_iter(self) -> Iterator[int]:
        """Iterator yielding task indices."""
        yield from range(len(self.tasks))

    def num_tasks(self) -> int | None:
        """Total number of tasks of dataset. Returns None if the number of tasks is infinite."""
        return len(self.tasks)


def split_data(
    ixs: set[int],
    dataset: TransformDataset,
    train_samples: int | None,
    test_samples: int,
    check_for_overlap: bool = False,
) -> tuple[list[int], list[int]]:
    """Split data into train and test indices."""

    if test_samples > len(ixs):
        raise ValueError(f"Not enough data for generating test samples. (ixs = {ixs})")
    test_ixs = set(random.sample(list(ixs), k=test_samples))

    train_ixs = ixs - set(test_ixs)
    if check_for_overlap:
        train_ixs = {
            ix for ix in train_ixs if not any(dataset.overlaps(ix, ix2) for ix2 in test_ixs)
        }

    if not train_ixs:
        raise ValueError("Not enough data for generating any train samples.")

    if train_samples is not None:
        if train_samples > len(train_ixs):
            logger.debug(
                f"Not enough data for generating train samples. "
                f"Reduced to available samples. (ixs= {train_ixs})"
            )
            train_samples = len(train_ixs)
        train_ixs = set(random.sample(list(train_ixs), k=train_samples))

    return sorted(train_ixs), sorted(test_ixs)


class PrecomputedTaskDataset(TaskDataset[TaskIx]):
    """Wrapper for generic task dataset for fetching task data in parallel.

    Args:
        task_ds: Task dataset to wrap
        task_size: Number of data points to fetch for each task.
            If None, fetch all available data.
    """

    def __init__(self, task_ds: TaskDataset[TaskIx], task_size: int | None = None):
        self.task_ds = task_ds
        self.task_size = task_size

    def __getitem__(self, ix: TaskIx) -> Task | None:
        task = self.task_ds[ix]
        if task is None:
            return None
        train_set = _precompute_data(task.train_set, size=self.task_size, shuffle=True)
        assert task.test_set is not None
        test_set = _precompute_data(task.test_set, shuffle=False)

        precomputed_task = Task(
            task_id=task.task_id,
            train_set=train_set,
            test_set=test_set,
            loss_fn=task.loss_fn,
            metrics=task.metrics,
            num_classes=task.num_classes,
        )
        return precomputed_task

    def task_iter(self) -> Iterator[TaskIx]:
        """Iterator yielding task indices."""
        return self.task_ds.task_iter()

    def num_tasks(self) -> int | None:
        """Total number of tasks of dataset. Returns None if the number of tasks is infinite."""
        return self.task_ds.num_tasks()


def _precompute_data(
    ds: TransformDataset, size: int | None = None, shuffle: bool = False
) -> TransformDataset:
    if not size:
        size = len(ds)
    size = min(size, len(ds))
    if shuffle:
        idx_iter: Iterable[int] = random.sample(range(len(ds)), k=size)
    else:
        idx_iter = itertools.islice(range(len(ds)), size)
    data = [ds[ix] for ix in idx_iter]
    precomputed_ds = TensorDataset(data)

    return TransformDataset(
        dataset=precomputed_ds,
        data_transforms=None,
        target_transforms=None,
        polygons=ds.polygons,
    )
