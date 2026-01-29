from copy import deepcopy
from typing import Any

import pytest
import torch
from torch.autograd import grad

from eurocropsmeta.algorithms.maml import MAML, MAMLConfig
from eurocropsmeta.dataset.task import Task
from eurocropsmeta.models.base import ModelBuilder

from .conftest import NUM_CLASSES

CONFIG_PARAMS = [
    {
        "reset_head": reset_head,
        "first_order": first_order,
        "meta_lr": meta_lr,
    }
    for reset_head in [False, True]
    for first_order in [False, True]
    for meta_lr in [0.0, 0.1]
]


@pytest.fixture(params=CONFIG_PARAMS)
def config(request: Any) -> MAMLConfig:
    num_classes = NUM_CLASSES if not request.param["reset_head"] else None

    return MAMLConfig(
        inner_lr=0.5,
        outer_lr=0.1,
        tasks_per_batch=2,
        total_task_batches=3,
        train_adaption_steps=2,
        num_classes=num_classes,
        **request.param,
    )


@pytest.fixture
def maml(config: MAMLConfig, model_builder: ModelBuilder) -> MAML:
    return MAML(config=config, model_builder=model_builder)


def test_maml_inner_update(maml: MAML, tasks: list[Task]) -> None:
    task = tasks[0]
    data, target = next(iter(task.train_dl(batch_size=10)))

    model = maml.model
    maml.model_builder.reset_head(model, task.num_classes)
    data = data.to(model.device)
    target = target.to(model.device)
    model_copy = deepcopy(model)

    loss = task.loss_fn(model(data), target)
    loss2 = task.loss_fn(model_copy(data), target)
    grads = grad(loss2, list(model_copy.parameters()), create_graph=False)

    maml.inner_update(model, loss, inner_step=0)
    for p1, p2, g in zip(model.parameters(), model_copy.parameters(), grads):
        expected = p2 - maml.config.inner_lr * g
        assert torch.allclose(p1, expected)


def test_maml_adapt(maml: MAML, tasks: list[Task]) -> None:
    old_model = deepcopy(maml.model)
    maml.adapt(tasks)
    for p1, p2 in zip(maml.model.backbone.parameters(), old_model.backbone.parameters()):
        assert not (p1 == p2).all()

    for p1, p2 in zip(maml.model.head.parameters(), old_model.head.parameters()):
        assert (p1 == p2).all() == maml.config.reset_head

    inner_lr = maml.inner_learner.lr.item()
    learn_inner_lr = maml.inner_learner.learn_lr
    assert (inner_lr == maml.config.inner_lr) != learn_inner_lr
    assert learn_inner_lr == (maml.config.meta_lr != 0.0)


def test_maml_eval(maml: MAML, tasks: list[Task]) -> None:
    old_model = deepcopy(maml.model)
    task = tasks[0]
    inner_lr = maml.inner_learner.lr.item()

    loss, metrics, task_model = maml.eval(task)

    assert {metric.name for metric in metrics} == {metric.name for metric in task.metrics}
    for p1, p2 in zip(maml.model.parameters(), old_model.parameters()):
        assert (p1 == p2).all()

    for p1, p2 in zip(maml.model.parameters(), task_model.parameters()):
        assert not (p1 == p2).all()

    assert inner_lr == maml.inner_learner.lr.item()
