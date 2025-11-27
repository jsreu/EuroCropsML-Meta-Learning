from copy import deepcopy
from typing import Any

import pytest
import torch
from torch.autograd import grad

from eurocropsmeta.algorithms.anil import ANIL, ANILConfig
from eurocropsmeta.dataset.task import Task
from eurocropsmeta.models.base import ModelBuilder

from .conftest import NUM_CLASSES

CONFIG_PARAMS = [
    {"reset_head": reset_head, "meta_lr": meta_lr}
    for reset_head in [False, True]
    for meta_lr in [0.0, 0.1]
]


@pytest.fixture(params=CONFIG_PARAMS)
def config(request: Any) -> ANILConfig:
    num_classes = NUM_CLASSES if not request.param["reset_head"] else None

    return ANILConfig(
        inner_lr=0.5,
        outer_lr=0.1,
        tasks_per_batch=2,
        total_task_batches=3,
        train_adaption_steps=2,
        num_classes=num_classes,
        **request.param,
    )


@pytest.fixture
def anil(config: ANILConfig, model_builder: ModelBuilder) -> ANIL:
    return ANIL(config=config, model_builder=model_builder)


def test_anil_inner_update(anil: ANIL, tasks: list[Task]) -> None:
    task = tasks[0]
    data, target = next(iter(task.train_dl(batch_size=10)))

    model = anil.model
    anil.model_builder.reset_head(model, task.num_classes)
    data = data.to(model.device)
    target = target.to(model.device)
    model_copy = deepcopy(model)

    model_copy = deepcopy(model)

    loss = task.loss_fn(model(data), target)
    loss2 = task.loss_fn(model_copy(data), target)
    grads = grad(loss2, list(model_copy.head.parameters()), create_graph=False)

    anil.inner_update(model, loss, inner_step=0)
    for p1, p2, g in zip(model.head.parameters(), model_copy.head.parameters(), grads):
        expected = p2 - anil.config.inner_lr * g
        assert torch.allclose(p1, expected)

    for p1, p2 in zip(model.backbone.parameters(), model_copy.backbone.parameters()):
        assert (p1 == p2).all()


def test_anil_adapt(anil: ANIL, tasks: list[Task]) -> None:
    old_model = deepcopy(anil.model)
    anil.adapt(tasks)
    for p1, p2 in zip(anil.model.backbone.parameters(), old_model.backbone.parameters()):
        assert not (p1 == p2).all()

    for p1, p2 in zip(anil.model.head.parameters(), old_model.head.parameters()):
        assert (p1 == p2).all() == anil.config.reset_head

    inner_lr = anil.inner_learner.lr.item()
    learn_inner_lr = anil.inner_learner.learn_lr
    assert (inner_lr == anil.config.inner_lr) != learn_inner_lr
    assert learn_inner_lr == (anil.config.meta_lr != 0.0)


def test_anil_eval(anil: ANIL, tasks: list[Task]) -> None:
    old_model = deepcopy(anil.model)
    task = tasks[0]
    inner_lr = anil.inner_learner.lr.item()

    loss, metrics, task_model = anil.eval(task)

    assert {metric.name for metric in metrics} == {metric.name for metric in task.metrics}

    for p1, p2 in zip(anil.model.parameters(), old_model.parameters()):
        assert (p1 == p2).all()

    for p1, p2 in zip(anil.model.backbone.parameters(), task_model.backbone.parameters()):
        assert (p1 == p2).all()

    for p1, p2 in zip(anil.model.head.parameters(), task_model.head.parameters()):
        assert not (p1 == p2).all()

    assert inner_lr == anil.inner_learner.lr.item()
