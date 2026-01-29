from copy import deepcopy
from typing import Any

import pytest
import torch
from torch.autograd import grad

from eurocropsmeta.algorithms.alfa import ALFA, ALFAConfig
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
    for meta_lr in [
        1.0,
        0.1,
    ]
]


@pytest.fixture(params=CONFIG_PARAMS)
def config(request: Any) -> ALFAConfig:
    num_classes = NUM_CLASSES if not request.param["reset_head"] else None

    return ALFAConfig(
        outer_lr=0.1,
        tasks_per_batch=2,
        total_task_batches=3,
        train_adaption_steps=2,
        layer_depth=1,
        num_classes=num_classes,
        **request.param,
    )


@pytest.fixture
def alfa(config: ALFAConfig, model_builder: ModelBuilder) -> ALFA:
    return ALFA(config=config, model_builder=model_builder)


def test_alfa_inner_update(alfa: ALFA, tasks: list[Task]) -> None:
    task = tasks[0]
    data, target = next(iter(task.train_dl(batch_size=10)))

    alfa.inner_learner.record_params = True  # type: ignore[assignment]
    model = alfa.model
    alfa.model_builder.reset_head(model, task.num_classes)
    data = data.to(model.device)
    target = target.to(model.device)

    model_copy = deepcopy(model)

    loss = task.loss_fn(model(data), target)
    loss2 = task.loss_fn(model_copy(data), target)
    grads = grad(loss2, list(model_copy.parameters()), create_graph=False)
    backbone_grads = grads[:-2]
    head_grads = grads[-2:]

    alfa.inner_update(model, loss, inner_step=0)
    assert alfa.inner_learner.recorded_params
    recorded_params: list[dict[str, dict[str, float]]] = alfa.inner_learner.recorded_params
    for p1, p2, g in zip(
        model.backbone.parameters(), model_copy.backbone.parameters(), backbone_grads
    ):
        expected = (
            recorded_params[0]["backbone"]["beta"] * p2
            - recorded_params[0]["backbone"]["alpha"] * g
        )
        assert torch.allclose(p1, expected)

    for p1, p2, g in zip(model.head.parameters(), model_copy.head.parameters(), head_grads):
        expected = recorded_params[0]["head"]["beta"] * p2 - recorded_params[0]["head"]["alpha"] * g
        assert torch.allclose(p1, expected)


def test_alfa_adapt(alfa: ALFA, tasks: list[Task]) -> None:
    old_model = deepcopy(alfa.model)
    old_learner = deepcopy(alfa.inner_learner)
    alfa.adapt(tasks)
    for p1, p2 in zip(alfa.model.backbone.parameters(), old_model.backbone.parameters()):
        assert not (p1 == p2).all()

    for p1, p2 in zip(alfa.model.head.parameters(), old_model.head.parameters()):
        assert (p1 == p2).all() == alfa.config.reset_head

    assert not all(
        (p1 == p2).all()
        for p1, p2 in zip(alfa.inner_learner.parameters(), old_learner.parameters())
    )


def test_alfa_eval(alfa: ALFA, tasks: list[Task]) -> None:
    old_model = deepcopy(alfa.model)
    task = tasks[0]

    loss, metrics, task_model = alfa.eval(task)

    assert {metric.name for metric in metrics} == {metric.name for metric in task.metrics}

    for p1, p2 in zip(alfa.model.parameters(), old_model.parameters()):
        assert (p1 == p2).all()

    for p1, p2 in zip(alfa.model.parameters(), task_model.parameters()):
        assert not (p1 == p2).all()
