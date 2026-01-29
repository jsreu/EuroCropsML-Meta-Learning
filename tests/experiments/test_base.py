from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from eurocropsmeta.experiment.base import TunedExperiment
from eurocropsmeta.experiment.runs import RunResult, get_run_results
from eurocropsmeta.experiment.utils import TuningConfig, get_timestamp


@pytest.mark.parametrize("study_name", ["mock_run", None])
def test_tuned_experiment_run_tuning(
    study_name: str | None,
    mock_experiment: TunedExperiment,
    mlflow_logger_mock: MagicMock,
) -> None:
    mock_results = [0.001, 0.01, 0.1]
    tuning_params = TuningConfig(
        floats={},
        ints={},
        categoricals={"mock_result": mock_results},
    )
    tuning_result = mock_experiment.run_tuning(study_name=study_name, tuning_params=tuning_params)

    assert tuning_result.metric_value == max(mock_results)
    loaded_run_results = {
        result.run_name: result
        for result in get_run_results(mock_experiment.runs_dir, prefix="tuning")
    }

    for n, call_arg in enumerate(mlflow_logger_mock.log_json.call_args_list):
        logged_result = RunResult(**call_arg.args[1])
        assert logged_result == loaded_run_results[logged_result.run_name]

        assert mlflow_logger_mock.start_run.call_args_list[n].args == (logged_result.run_name,)
        assert mlflow_logger_mock.end_run.call_args_list[n].args == ()


@pytest.mark.parametrize("run_name", ["mock_run", None])
def test_tuned_experiment_run_training(
    run_name: str | None,
    mock_experiment: TunedExperiment,
    mlflow_logger_mock: MagicMock,
) -> None:
    model_artifact_uri = "mock_artifact_uri"
    mlflow_logger_mock.get_artifact_uri.return_value = model_artifact_uri
    extra_params = {"mock_result": 1.0}
    training_result = mock_experiment.run_training(run_name=run_name, extra_params=extra_params)

    assert training_result.metric_value == extra_params["mock_result"]
    assert training_result.model_artifact_uri == model_artifact_uri
    loaded_run_results = get_run_results(runs_dir=mock_experiment.runs_dir, prefix="training")

    assert len(loaded_run_results) == 1
    assert loaded_run_results[0].model_dump() == training_result.model_dump()

    mlflow_logger_mock.log_json.assert_called_once()
    logged_result = RunResult(**mlflow_logger_mock.log_json.call_args.args[1])
    assert logged_result.model_dump() == training_result.model_dump()

    mlflow_logger_mock.start_run.assert_called_once_with(training_result.run_name)
    mlflow_logger_mock.end_run.assert_called_once_with()


@pytest.mark.parametrize("run_name", ["mock_run", None])
def test_tuned_experiment_load_run_results(
    run_name: str | None,
    mock_experiment: TunedExperiment,
    mocker: MockerFixture,
) -> None:
    if run_name is None:
        run_name = get_timestamp()
    run_result = RunResult(
        run_name=run_name,
        key_metric=mock_experiment.config.key_metric,
        metric_value=1.0,
        params={"mock_result": 1},
        config={name: config.model_dump() for name, config in mock_experiment.run_config().items()},
    )
    run_result2 = deepcopy(run_result)
    run_result2.config["model_config"]["layers"] = 2

    run_result3 = deepcopy(run_result)
    run_result3.run_name = "same_config_diff_name"

    mocker.patch(
        "eurocropsmeta.experiment.base.get_run_results",
        return_value=[run_result, run_result2, run_result3],
    )
    results = mock_experiment.load_run_results(run_name=run_name, tuning=False)

    # Models are incompatible or have different names, so only one result is expected
    assert len(results) == 1
    assert results[0] == run_result
