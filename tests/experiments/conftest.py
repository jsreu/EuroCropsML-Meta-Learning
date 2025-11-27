from pathlib import Path
from typing import Any, Mapping, cast
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from eurocropsmeta.experiment.base import ExperimentConfig, TunedExperiment
from eurocropsmeta.experiment.utils import TuningConfig
from eurocropsmeta.models.base import ModelConfig
from eurocropsmeta.train.callback import TrainCallback
from eurocropsmeta.train.utils import ScalarMetric
from eurocropsmeta.utils import BaseConfig


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    runs_dir = tmp_path.joinpath("runs")
    runs_dir.mkdir()
    return runs_dir


class MockModelConfig(ModelConfig):
    model_builder: str = "MockBuilder"
    layers: int = 1


@pytest.fixture
def model_config() -> ModelConfig:
    return MockModelConfig()  #


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    return tmp_path / "experiment"


@pytest.fixture
def mlflow_logger_mock(mocker: MockerFixture) -> MagicMock:
    logger_mock: MagicMock = mocker.MagicMock()
    mocker.patch("eurocropsmeta.experiment.base.MLFlowLogger", return_value=logger_mock)
    return logger_mock


class MockTrainConfig(BaseConfig):
    mock_result: float = 1.0


class MockMetric(ScalarMetric):
    def __init__(self, mock_result: float = 1.0, name: str = "Accuracy"):
        self.mock_result = mock_result
        self.name = name

    def set_scalar(self, val: float) -> None:
        self.mock_result = val

    def get_scalar(self) -> float:
        return self.mock_result


class MockExperimentConfig(ExperimentConfig):
    train_config: MockTrainConfig


class MockExperiment(TunedExperiment):
    def train_model(
        self, run_config: Mapping[str, BaseConfig], callbacks: list[TrainCallback]
    ) -> None:
        mock_config = cast(MockTrainConfig, run_config["mock_config"])
        for callback in callbacks:
            val_loss = 0.1  # mock value, has no effect
            mock_metric = MockMetric()
            mock_metric.set_scalar(mock_config.mock_result)
            val_metrics = [mock_metric]
            callback.validation_callback(
                val_loss,
                val_metrics,
                model=MagicMock(),
                step=1,
            )

    def run_config(self) -> Mapping[str, BaseConfig]:
        return {
            "mock_config": self.config.train_config,
            "model_config": self.model_config,
        }

    def hyperparameters(self, run_config: Mapping[str, BaseConfig]) -> dict[str, Any]:
        return run_config["mock_config"].model_dump()


@pytest.fixture
def mock_experiment(
    model_config: ModelConfig,
    experiment_dir: Path,
) -> TunedExperiment:

    config = MockExperimentConfig(
        name="mock",
        key_metric="Accuracy",
        tuning_params=TuningConfig(
            floats={},
            ints={},
            categoricals={"value": [1, 2, 3]},
        ),
        validate_every=1,
        train_config=MockTrainConfig(),
    )
    experiment = MockExperiment(config, model_config=model_config, experiment_dir=experiment_dir)

    return experiment
