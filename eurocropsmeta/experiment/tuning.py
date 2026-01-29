from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import optuna
import torch
from optuna import Study
from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler

from eurocropsmeta.experiment.utils import TuningConfig
from eurocropsmeta.settings import Settings

from .runs import RunResult

logger = logging.getLogger(__name__)


Trainable = Callable[[str, dict[str, Any], optuna.Trial | None], RunResult]


def _objective(
    trial: optuna.Trial,
    trainable: Trainable,
    study_name: str,
    tuning_params: TuningConfig,
) -> float:
    extra_params: dict[str, float | int | bool | str] = {}
    for name, float_params in tuning_params.floats.items():
        extra_params[name] = trial.suggest_float(
            name, log=float_params.log, low=float_params.low, high=float_params.high
        )
    for name, int_params in tuning_params.ints.items():
        extra_params[name] = trial.suggest_int(
            name, log=int_params.log, low=int_params.low, high=int_params.high
        )
    for name, cat_params in tuning_params.categoricals.items():
        extra_params[name] = trial.suggest_categorical(name, cat_params)

    run_name = study_name + "-trial-" + str(trial._trial_id)
    logger.info("Starting new tuning run (%s)", run_name)
    logger.info("Tuning params: %s", extra_params)

    run_result = trainable(run_name, extra_params, trial)
    return run_result.metric_value


def run_tuning(
    trainable: Trainable,
    tuning_params: TuningConfig,
    study_name: str,
    runs_dir: Path,
    num_workers: int | None = None,
    num_trials: int | None = None,
    tuning_sampler: str | None = None,
    tuning_pruning_percentile: float | None = None,
    tuning_pruning_warmup_steps: int | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> Study:
    """Tune trainable function with given tuning params.

    Args:
        trainable: The function to optimize.
            Should take dictionary of a run name and tuning_params and optionally a tuning trial
            as input and return the result of the training run.
        tuning_params: Nested dictionary describing the hyperparameter space to search.
        study_name: The name given to the optuna study.
        runs_dir: Directory where the runs are stored.
        num_workers: Number of parallel processes to perform tuning.
            If None, no parallelization is used.
        num_trials: Maximum number of trials to perform on each worker.
            If None, no upper bound is applied.
        tuning_sampler: The sampling strategy to use for suggesting parameter values during trials.
            If None, uses a GridSampler if the search space over tunable parameters is discrete,
            otherwise (there is at least one non-discrete parameter) uses a RandomSampler.
        tuning_pruning_percentile: Percentile rule used for hyperparameter tuning trial pruning.
            If None, no pruning of trials. Otherwise, set a value X between 0 and 100 to keep the
            top Xth percentile of trials running (i.e. 50 for median pruning rule).
        tuning_pruning_warmup_steps: Warmup rule used for hyperparameter tuning trial pruning.
            If None, no warmup. Otherwise, specifies a number of minimum training steps before
            the pruning of trials becomes active.
        direction: Direction of optimization.

    Returns:
        The optuna study object.

    Raises:
        ValueError: If the provided tuning_sampler does not match a known optuna sampling strategy.
    """

    sampler_seed = Settings().seed

    if tuning_sampler is None:
        if (
            len(tuning_params.floats.keys()) > 0 or len(tuning_params.ints.keys()) > 0
        ):  # use RandomSampler as default for non-categoricals
            sampler: BaseSampler = RandomSampler(seed=sampler_seed)
        else:  # use GridSampler as default for only categoricals
            sampler = GridSampler(search_space=tuning_params.categoricals, seed=sampler_seed)
    elif tuning_sampler.lower() == "grid":
        if len(tuning_params.floats.keys()) > 0 or len(tuning_params.ints.keys()) > 0:
            logger.warn(
                "Non-categorical parameters were specified in combination with a GridSampling "
                "strategy. GridSampler only handles categorical parameters, all other "
                "parameters will be ignored."
            )
        sampler = GridSampler(search_space=tuning_params.categoricals, seed=sampler_seed)
    elif tuning_sampler.lower() == "random":
        sampler = RandomSampler(seed=sampler_seed)
    elif tuning_sampler.lower() == "tpe":
        sampler = TPESampler(seed=sampler_seed, constant_liar=True)
    else:
        raise ValueError(
            f"Unknown sampling strategy {tuning_sampler}, expected 'grid' or 'random'."
        )

    if tuning_pruning_percentile is None:
        tuning_pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()
    else:
        tuning_pruner = optuna.pruners.PercentilePruner(
            tuning_pruning_percentile,
            n_startup_trials=1,  # do at least one full reference run before pruning
            n_warmup_steps=tuning_pruning_warmup_steps or 1,
        )

    storage_path = runs_dir.joinpath(f"{study_name}.db")
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction=direction,
        storage=f"sqlite:///{storage_path}",
        pruner=tuning_pruner,
    )
    try:
        if num_workers is None or 1 >= num_workers >= 0:
            study.optimize(
                partial(
                    _objective,
                    trainable=trainable,
                    study_name=study_name,
                    tuning_params=tuning_params,
                ),
                n_trials=num_trials,
            )
        else:
            logger.info(
                f"Starting trial runs in multiple processes using {num_workers} workers. "
                "This requires copying data to individual processes and might take a while."
            )
            context = torch.multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=study.optimize,
                    args=(
                        partial(
                            _objective,
                            trainable=trainable,
                            study_name=study_name,
                            tuning_params=tuning_params,
                        ),
                    ),
                    kwargs={"n_trials": num_trials},
                )
                for _ in range(num_workers)
            ]
            for p in processes:
                logger.info("Starting a new worker.")
                p.start()
                sampler.reseed_rng()  # avoids duplicate trials
            for p in processes:
                p.join()
    except KeyboardInterrupt:
        logger.warning("Encountered keyboard interrupt. Stopping hyperparameter tuning.")
        # Clean up database if study is interrupted to make sure
        # runs with same name can be started.
        storage_path.unlink()
    return study
