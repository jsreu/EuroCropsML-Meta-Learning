import json
from pathlib import Path
from typing import Any

import optuna
import pytest

from eurocropsmeta.experiment.runs import RunResult
from eurocropsmeta.experiment.tuning import run_tuning
from eurocropsmeta.experiment.utils import TuningConfig

KEY_METRIC = "Acc"


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir


def _build_run_result(run_name: str, value: float = 1.0) -> RunResult:
    params = {"value": value}
    return RunResult(
        run_name=run_name,
        key_metric=KEY_METRIC,
        metric_value=value,
        params=params,
    )


@pytest.mark.parametrize("run_name", ["test", None])
def test_run_tuning(runs_dir: Path, run_name: str | None) -> None:
    def trainable(
        run_name: str, extra_params: dict[str, Any], trial: optuna.Trial | None = None
    ) -> RunResult:
        result = _build_run_result(run_name, extra_params["value"])
        with open(runs_dir / f"{run_name}.json", "w") as f:
            f.write(result.model_dump_json())
        return result

    tuning_params = TuningConfig(
        floats={},
        ints={},
        categoricals={"value": list(range(10))},
    )
    study_name = "test-study"
    study = run_tuning(
        trainable=trainable,
        tuning_params=tuning_params,
        study_name=study_name,
        runs_dir=runs_dir,
    )

    assert len(study.trials) == len(tuning_params.categoricals["value"])
    assert study.best_trial.value == max(tuning_params.categoricals["value"])

    run_results = list(runs_dir.glob("*.json"))
    assert len(run_results) == len(tuning_params.categoricals["value"])

    results = [RunResult(**json.loads(result_path.read_text())) for result_path in run_results]
    assert {result.params["value"] for result in results} == set(
        tuning_params.categoricals["value"]
    )
    assert {result.metric_value for result in results} == set(tuning_params.categoricals["value"])
    assert all(result.run_name.startswith(study_name) for result in results)
