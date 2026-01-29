from copy import deepcopy
from datetime import datetime
from typing import Any, Sequence, Type, TypeVar, cast

import mlflow
from pydantic import BaseModel, field_validator

import eurocropsmeta.algorithms
import eurocropsmeta.models.transformer
from eurocropsmeta.algorithms.base import MetaLearnAlgorithm, MetaLearnAlgorithmConfig
from eurocropsmeta.models.base import ModelBuilder, ModelConfig
from eurocropsmeta.settings import Settings
from eurocropsmeta.utils import BaseConfig

TIMESTAMP_FORMAT = "%Y%m%d-%H-%M-%S"


class DistributionalParamConfig(BaseModel):
    """Config for a single distributional search space."""

    log: bool
    low: float | int
    high: float | int


class DistributionalFloatParamConfig(DistributionalParamConfig):
    """Config for a single distributional continuum search space."""

    low: float
    high: float


class DistributionalIntParamConfig(DistributionalParamConfig):
    """Config for a single distributional integer search space."""

    low: int
    high: int


class TuningConfig(BaseModel):
    """Config for collected hyperparameter search space."""

    floats: dict[str, DistributionalFloatParamConfig]
    ints: dict[str, DistributionalIntParamConfig]
    categoricals: dict[str, Sequence[float] | Sequence[int] | Sequence[bool] | Sequence[str]]

    @field_validator("categoricals")
    @classmethod
    def validate_categoricals(
        cls,
        v: dict[str, Sequence[float] | Sequence[int] | Sequence[bool] | Sequence[str]],
    ) -> dict[str, Sequence[float] | Sequence[int] | Sequence[bool] | Sequence[str]]:
        """Ensure correct typing of integer categorical parameters."""
        for key, value in v.items():
            if all(isinstance(item, (int, float)) and int(item) == item for item in value):
                v[key] = list(map(int, value))
        return v


def get_meta_learner(
    meta_config: MetaLearnAlgorithmConfig, model_builder: ModelBuilder
) -> MetaLearnAlgorithm:
    """Get meta learner for given config."""
    algorithm_class = getattr(
        eurocropsmeta.algorithms,
        meta_config.name,
    )
    return cast(
        MetaLearnAlgorithm,
        algorithm_class(config=meta_config, model_builder=model_builder),
    )


def get_model_builder(model_config: ModelConfig) -> ModelBuilder:
    """Get model builder for given config."""
    model_builder_class: Type[ModelBuilder] = getattr(
        eurocropsmeta.models.transformer, model_config.model_builder
    )
    return model_builder_class(model_config)


BaseConfigT = TypeVar("BaseConfigT", bound=BaseConfig)


def overwrite_config(config: BaseConfigT, extra_params: dict[str, Any]) -> BaseConfigT:
    """Overwrite fields in config given by extra_params."""
    config = deepcopy(config)
    for name, param in extra_params.items():
        if name in config.params():
            setattr(config, name, param)

    return config


def recursive_update(d: dict[str, Any], key: str, value: Any) -> None:
    """Recursively update dict with given key and value."""
    if key in d:
        d[key] = value
    else:
        for item in d.values():
            if isinstance(item, dict):
                recursive_update(item, key, value)


def configure_mlflow(experiment_name: str | None = None) -> None:
    """Configure mlflow tracking server."""
    mlflow.set_tracking_uri(Settings().mlflow_uri)
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)


def get_timestamp() -> str:
    """Get timestamp for current date and time."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)
