import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pytest

from eurocropsmeta.experiment.runs import RunResult, get_run_results
from eurocropsmeta.models.base import ModelConfig


@pytest.fixture()
def run_result(model_config: ModelConfig) -> RunResult:
    return RunResult(
        run_name="test",
        key_metric="Acc",
        metric_value=1.0,
        params={"epochs": 1, "lr": 0.1},
        config={
            "model_config": model_config.model_dump(),
            "train_config": {"lr": 1.0, "epochs": 3},
        },
    )


@pytest.fixture
def run_results(run_result: RunResult) -> list[RunResult]:
    results = []
    for n in range(10):
        result = deepcopy(run_result)
        result.run_name = f"{result.run_name}_{n}"
        result.metric_value = n
        results.append(result)
    return results


def test_run_result_override_config(run_result: RunResult) -> None:
    assert run_result.config["train_config"]["lr"] == run_result.params["lr"]


def test_run_result_ignore_nontuning_params(run_result: RunResult) -> None:
    assert run_result.params == {"lr": 0.1}
    assert run_result.config["train_config"]["epochs"] == 3


@pytest.mark.parametrize("prefix", ["tuning", "training"])
def test_get_run_results(
    runs_dir: Path,
    run_results: list[RunResult],
    prefix: Literal["tuning", "training"],
) -> None:
    for result in run_results:
        result_path = runs_dir.joinpath(f"{prefix}-{result.run_name}.json")
        with open(result_path, "w") as f:
            json.dump(result.model_dump(), f)

    loaded_results = get_run_results(runs_dir, prefix=prefix)
    assert len(loaded_results) == len(run_results)
    for result, loaded_result in zip(loaded_results, run_results[::-1]):
        assert result == loaded_result

    other_prefix = "tuning" if prefix == "training" else "training"
    assert not get_run_results(runs_dir, prefix=other_prefix)  # type: ignore[arg-type]
