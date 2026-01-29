from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Type
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel
from typer.testing import CliRunner, Typer

from eurocropsmeta.experiment.cli import (
    BuilderT,
    ConfigT,
    build_experiment_app,
    eurocrops_app,
)
from eurocropsmeta.experiment.eurocrops.config import EuroCropsTransferConfig
from eurocropsmeta.experiment.transfer import (
    TransferExperiment,
    TransferExperimentBuilder,
)


class MockConfig(BaseModel):
    name: str
    value: int


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config(monkeypatch: Any, tmp_path: Path) -> MockConfig:
    config = MockConfig(name="test_experiment", value=1)

    experiment_path = tmp_path / "experiment"
    config_path = experiment_path / "test_experiment" / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config.model_dump()))

    monkeypatch.setenv("EUROCROPS_META_LEARNING_EXPERIMENT_DIR", str(experiment_path.absolute()))
    return config


@pytest.fixture
def experiment_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def builder_class(experiment_mock: MagicMock) -> Type[BuilderT]:
    class MockBuilder(TransferExperimentBuilder):
        def __init__(self, config: ConfigT) -> None:
            super().__init__(config)
            pass

        def build_experiment(self, mode: Literal["pretrain", "finetune"]) -> TransferExperiment:
            return experiment_mock

    return MockBuilder  # type: ignore[return-value]


@pytest.fixture
def app(config: MockConfig, builder_class: Type[BuilderT]) -> Typer:
    return build_experiment_app(
        experiment_name=config.name,
        config_class=MockConfig,
        builder_class=builder_class,
    )


def test_print_config(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config"], catch_exceptions=False)
    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()


def test_print_config_overrides(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config", "value=2"], catch_exceptions=False)
    config.value = 2
    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()


def test_pretraining(app: Typer, runner: CliRunner, experiment_mock: MagicMock) -> None:
    result = runner.invoke(app, ["pretraining"], catch_exceptions=False)
    assert result.exit_code == 0

    experiment_mock.run_pretraining.assert_called_once()


def test_pretrain_tuning(app: Typer, runner: CliRunner, experiment_mock: MagicMock) -> None:
    result = runner.invoke(app, ["pretrain-tuning"], catch_exceptions=False)
    assert result.exit_code == 0

    experiment_mock.run_pretrain_tuning.assert_called_once()


@pytest.mark.parametrize("pretrain_run", ["auto", "random", "test_run"])
def test_finetuning(
    app: Typer,
    runner: CliRunner,
    pretrain_run: str,
    experiment_mock: MagicMock,
) -> None:
    task = "test_task"
    args = ["finetuning", task, "--pretrain-run", pretrain_run]
    result = runner.invoke(app, args, catch_exceptions=False)
    assert result.exit_code == 0

    pretrain_run_name = None if pretrain_run == "random" else pretrain_run
    experiment_mock.run_finetuning.assert_called_once_with(
        task_name=task, run_name=None, pretrain_run_name=pretrain_run_name
    )


@pytest.mark.parametrize("pretrain_run", ["auto", "random", "test_run"])
def test_finetune_tuning(
    app: Typer, runner: CliRunner, pretrain_run: str, experiment_mock: MagicMock
) -> None:
    task = "test_task"
    args = ["finetune-tuning", task, "--pretrain-run", pretrain_run]
    result = runner.invoke(app, args, catch_exceptions=False)
    assert result.exit_code == 0

    pretrain_run_name = None if pretrain_run == "random" else pretrain_run
    experiment_mock.run_finetune_tuning.assert_called_once_with(
        task_name=task, study_name=None, pretrain_run_name=pretrain_run_name
    )


EUROCROPS_OVERRIDES = [
    [],
    ["model=transformer_small"],
    ["model=transformer_big"],
    ["pretrain=pretrain"],
    ["pretrain=meta_20_5"],
    ["pretrain=meta_10_10.yaml"],
    ["pretrain=meta_10_1.yaml"],
    ["pretrain=meta_4_10.yaml"],
    ["pretrain=meta_4_1.yaml"],
    ["finetune=head"],
    ["finetune=headbackbonediff"],
    ["finetune=headbackbonesame"],
    ["pretrain=meta_20_5", "pretrain.meta_config.name=MAML"],
]

EXPERIMENT_PARAMS = [
    (eurocrops_app, EuroCropsTransferConfig, overrides) for overrides in EUROCROPS_OVERRIDES
]


@pytest.mark.parametrize(
    "experiment_app,config_class,overrides",
    EXPERIMENT_PARAMS,
)
def test_experiment_config(
    runner: CliRunner,
    experiment_app: Typer,
    config_class: Type[ConfigT],
    overrides: list[str],
) -> None:
    args = ["config"] + overrides
    result = runner.invoke(experiment_app, args, catch_exceptions=False)
    assert result.exit_code == 0
    printed_config = result.stdout
    parsed_config = OmegaConf.to_object(OmegaConf.create(printed_config))
    assert config_class(**parsed_config)  # type: ignore[arg-type]
