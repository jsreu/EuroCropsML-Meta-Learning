import itertools
import math
from collections import defaultdict
from itertools import islice

import pytest
import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset
from shapely.geometry import Polygon

from eurocropsmeta.dataset.task import Task, split_data

from ..utils import SineDataset, interval_tasks, wrappedinterval_tasks


def test_task_dl() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    batch_size = 8
    num_batches = -(-len(transform_dataset) // batch_size)

    train_dl = task.train_dl(batch_size=batch_size)
    train_batch = next(iter(train_dl))
    assert len(train_dl) == num_batches
    assert train_batch[0].data.size(0) == batch_size
    assert train_batch[1].data.size(0) == batch_size

    test_dl = task.train_dl(batch_size=batch_size)
    test_batch = next(iter(test_dl))
    assert len(test_dl) == num_batches
    assert test_batch[0].data.size(0) == batch_size
    assert test_batch[1].data.size(0) == batch_size

    with pytest.raises(ValueError):
        task.val_dl(batch_size=batch_size)


def test_task_dl_from_num_batches() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    num_batches = 5
    batch_size = len(transform_dataset) // num_batches

    train_dl = task.train_dl(num_batches=num_batches)
    train_batch = next(iter(train_dl))
    assert len(train_dl) == num_batches
    assert train_batch[0].data.size(0) == batch_size
    assert train_batch[1].data.size(0) == batch_size

    test_dl = task.train_dl(num_batches=num_batches)
    test_batch = next(iter(test_dl))
    assert len(test_dl) == num_batches
    assert test_batch[0].data.size(0) == batch_size
    assert test_batch[1].data.size(0) == batch_size

    with pytest.raises(ValueError):
        task.val_dl(num_batches=num_batches)


def test_task_dl_val_dl() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        val_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    batch_size = 8
    num_batches = -(-len(transform_dataset) // batch_size)

    val_dl = task.val_dl(batch_size=batch_size)
    val_samples, val_labels = next(iter(val_dl))
    assert len(val_dl) == num_batches
    assert val_samples.data.size(0) == batch_size
    assert val_labels.data.size(0) == batch_size


def test_task_dl_val_dl_from_num_batches() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        val_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    num_batches = 5
    batch_size = len(transform_dataset) // num_batches

    val_dl = task.val_dl(num_batches=num_batches)
    val_samples, val_labels = next(iter(val_dl))
    assert len(val_dl) == num_batches
    assert val_samples.data.size(0) == batch_size
    assert val_labels.data.size(0) == batch_size


def test_wrapclasstaskdataset_getitem() -> None:
    num_classes = 3
    total_classes = 4
    total_sup_classes = 5
    task_ds = wrappedinterval_tasks(
        total_sup_classes=total_sup_classes,
        total_classes=total_classes,
        num_classes=num_classes,
        sample=False,
    )
    task_ids = [
        (sc, combination)
        for sc in range(total_sup_classes)
        for combination in itertools.combinations(range(total_classes), r=num_classes)
    ]

    for task_ix in task_ids:
        task = task_ds[task_ix]
        assert task is not None
        assert task_ds._train_samples_per_class is not None
        assert len(task.train_set) == num_classes * task_ds._train_samples_per_class

        for n in range(len(task.train_set)):
            _, targets = task.train_set[n]
            assert (0 <= targets < num_classes).all()

        assert task.test_set is not None
        assert len(task.test_set) == num_classes * task_ds._test_samples_per_class

        for n in range(len(task.test_set)):
            _, targets = task.test_set[n]
            assert (0 <= targets < num_classes).all()

    assert task_ds[(0, (0, 1, 2, 2, 3))] is None
    assert task_ds[(0, (0, 2))] is None


def test_wrapclasstaskdataset_task_iter() -> None:
    num_classes = 3
    total_classes = 4
    total_sup_classes = 5
    task_ds = wrappedinterval_tasks(
        total_sup_classes=total_sup_classes,
        total_classes=total_classes,
        num_classes=num_classes,
        sample=False,
    )
    task_ids = list(task_ds.task_iter())
    task_dict = defaultdict(list)
    for sc, combination in task_ids:
        task_dict[sc].append(combination)
    expected = {
        sc: list(itertools.combinations(range(total_classes), r=num_classes))
        for sc in range(total_sup_classes)
    }
    assert expected == task_dict


def test_wrapclasstaskdataset_task_iter_sample() -> None:
    task_ds = wrappedinterval_tasks(
        total_sup_classes=5, total_classes=10, num_classes=3, sample=True
    )

    num_tasks = math.comb(10, 3)

    task_ids = list(islice(task_ds, num_tasks + 10))

    assert len(task_ids) == num_tasks + 10
    assert len(task_ids) > 1


def test_wrapclasstaskdataset_dataloader() -> None:
    task_ds = wrappedinterval_tasks(
        total_sup_classes=5, total_classes=10, num_classes=3, sample=True
    )

    batch_size = 10
    dl = task_ds.dataloader(batch_size=batch_size)
    batch = next(iter(dl))
    assert len(batch) == batch_size
    for task in batch:
        assert isinstance(task, Task)


def test_wrapclasstaskdataset_num_tasks() -> None:
    task_ds = wrappedinterval_tasks(
        total_sup_classes=5, total_classes=10, num_classes=3, sample=True
    )
    assert task_ds.num_tasks() is None

    task_ds = wrappedinterval_tasks(
        total_sup_classes=5, total_classes=10, num_classes=3, sample=False
    )
    assert task_ds.num_tasks() == 5 * (10 * 9 * 8) / (3 * 2)  # (5 times: 10 choose 3)


def test_classtaskdataset_task_iter() -> None:
    num_classes = 3
    total_classes = 10
    task_ds = interval_tasks(total_classes=total_classes, num_classes=num_classes, sample=False)
    task_ids = set(task_ds.task_iter())
    combinations = itertools.combinations(
        range(total_classes),
        r=num_classes,
    )
    assert all(task_id[0] == 0 for task_id in task_ids)
    assert {task_id[1] for task_id in task_ids} == set(combinations)


def test_classtaskdataset_task_iter_sample() -> None:
    task_ds = interval_tasks(total_classes=10, num_classes=3, sample=True)

    num_tasks = math.comb(10, 3)
    task_ids = list(islice(task_ds.task_iter(), num_tasks + 10))

    assert len(task_ids) == num_tasks + 10
    assert len(set(task_ids)) > 1
    assert len(set(task_ids)) <= num_tasks


def test_classtaskdataset_dataloader() -> None:
    task_ds = interval_tasks(total_classes=10, num_classes=3, sample=True)

    batch_size = 10
    dl = task_ds.dataloader(batch_size=batch_size)
    batch = next(iter(dl))
    assert len(batch) == batch_size
    for task in batch:
        assert isinstance(task, Task)


def test_classtaskdataset_num_tasks() -> None:
    task_ds = interval_tasks(total_classes=10, num_classes=3, sample=True)
    assert task_ds.num_tasks() is None

    task_ds = interval_tasks(total_classes=10, num_classes=3, sample=False)
    assert task_ds.num_tasks() == 10 * 9 * 8 / (3 * 2)  # (10 choose 3)


@pytest.mark.parametrize("train_samples", [5, None])
def test_split_data(train_samples: int | None) -> None:
    dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    ixs = set(range(10, 30))
    train_ix, test_ix = split_data(ixs, dataset, train_samples=train_samples, test_samples=10)
    assert all(ix in ixs for ix in train_ix)
    assert all(ix in ixs for ix in test_ix)
    assert not set(train_ix).intersection(test_ix)

    num_train = train_samples if train_samples is not None else 10
    assert len(test_ix) == 10
    assert len(train_ix) == min(num_train, 10)


def test_split_data_insufficient_data() -> None:
    dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    ixs = set(range(10, 30))

    # error if there is not enough test data
    with pytest.raises(ValueError):
        train_ix, test_ix = split_data(ixs, dataset, train_samples=None, test_samples=200)

    # error if there is barely enough test data but nothing left for training
    with pytest.raises(ValueError):
        train_ix, test_ix = split_data(ixs, dataset, train_samples=100, test_samples=20)

    # only debug warning if there is training data but fewer samples than specified
    train_ix, test_ix = split_data(ixs, dataset, train_samples=100, test_samples=10)
    assert len(test_ix) == 10
    assert len(train_ix) == len(ixs) - len(test_ix)


def test_split_data_check_for_overlap() -> None:
    polygons = {x: Polygon([(x, 0), (x + 1, 1), (x + 1, 0)]) for x in range(100)}
    dataset = TransformDataset(SineDataset(phase=0.5, size=100), polygons=polygons)
    ixs = set(range(10, 80))
    train_ix, test_ix = split_data(
        ixs, dataset, train_samples=None, test_samples=10, check_for_overlap=True
    )

    for ix1 in test_ix:
        assert not any(dataset.overlaps(ix1, ix2) for ix2 in train_ix)
