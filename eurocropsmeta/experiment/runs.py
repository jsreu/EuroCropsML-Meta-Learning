import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import mlflow
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core.core_schema import ValidationInfo

from eurocropsmeta.experiment.utils import configure_mlflow, recursive_update

logger = logging.getLogger(__name__)


NON_TUNING_PARAMS = [
    "epochs",
    "steps",
    "total_task_batches",
]


class RunResult(BaseModel):
    """Class for storing results of training/tuning runs."""

    run_name: str
    key_metric: str | dict[str, float]
    metric_value: float
    params: dict[str, Any]
    config: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    model_artifact_uri: str | None = None

    # suppress model_* protected namespace warnings
    model_config = ConfigDict(protected_namespaces=())

    @field_validator("config")
    @classmethod
    def override_config(cls, v: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        """Override other config with values from params."""
        params = info.data["params"]
        for name, param in params.items():
            recursive_update(v, name, param)
        return v

    @field_validator("params")
    @classmethod
    def ignore_nontuning_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Remove params which should not be considered tunable."""
        return {name: param for name, param in v.items() if name not in NON_TUNING_PARAMS}


def get_run_results(
    runs_dir: Path,
    prefix: Literal["tuning", "training"],
    direction: Literal["minimize", "maximize"] = "maximize",
) -> list[RunResult]:
    """Get results saved in given runs directory ordered by key metric."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"{runs_dir} is not a directory.")

    result_paths: Iterator[Path] = runs_dir.glob(f"{prefix}*.json")
    results = []
    for result_path in result_paths:
        with open(result_path) as f:
            result = RunResult(**json.load(f))
            results += [result]

    results.sort(key=lambda r: r.metric_value, reverse=(direction == "maximize"))
    return results


def get_mlflow_runs(experiments: list[str], filter_string: str) -> list[mlflow.entities.Run]:
    """Search experiments for runs matching filter string."""
    configure_mlflow()
    experiment_ids = [
        exp.experiment_id for exp in mlflow.search_experiments() if exp.name in experiments
    ]
    results: list[mlflow.entities.Run] = mlflow.search_runs(
        experiment_ids,
        filter_string=filter_string,
        output_format="list",
    )
    return results


def get_mflow_run_by_name(experiments: list[str], run_name: str) -> mlflow.entities.Run:
    """Get mlflow run by name from specified experiments."""
    filter_string = f'attributes.run_name LIKE "%{run_name}"'
    results = get_mlflow_runs(experiments, filter_string=filter_string)
    if len(results) != 1:
        filter_string = f'attributes.run_name LIKE "training-{run_name}"'
        results = get_mlflow_runs(experiments, filter_string=filter_string)
        if not len(results) == 1:
            raise ValueError(f"Found {len(results)} runs for run {run_name}. Expected 1.")
    return results[0]


def download_run_model(experiment_name: str, run_name: str, runs_dir: Path) -> None:
    """Download model weights for given run from mlflow backend.

    Saves the model checkpoint to <runs_dir>/<run_name>.
    """
    configure_mlflow()
    destination = runs_dir.joinpath(run_name)
    if destination.is_dir():
        logger.info("Model for run %s already downloaded.", run_name)
        return

    logger.info("Looking up run %s for experiment %s in mlflow.", run_name, experiment_name)
    run = get_mflow_run_by_name([experiment_name], run_name)
    logger.info("Downloading model weights from mlflow backend.")
    mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="model",
        dst_path=str(destination),
    )
