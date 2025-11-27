import json
import logging
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, TypeVar, cast

import optuna
import torch
from pydantic import BaseModel, ConfigDict

from eurocropsmeta.dataset.task import Task
from eurocropsmeta.experiment.logger import MLFlowCallback, MLFlowLogger
from eurocropsmeta.experiment.runs import RunResult, get_run_results
from eurocropsmeta.experiment.tuning import run_tuning
from eurocropsmeta.experiment.utils import (
    TuningConfig,
    get_model_builder,
    get_timestamp,
    overwrite_config,
)
from eurocropsmeta.models.base import Model, ModelConfig
from eurocropsmeta.settings import Settings
from eurocropsmeta.train.callback import (
    OptunaCallback,
    TrainCallback,
    ValidationCallback,
)
from eurocropsmeta.train.train import TrainConfig, Trainer
from eurocropsmeta.train.utils import set_seed
from eurocropsmeta.utils import BaseConfig

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """Config for training experiments."""

    name: str
    key_metric: str | dict[str, float]
    tuning_params: TuningConfig
    validate_every: int

    tags: dict[str, str] = {}
    tuning_workers: int | None = None
    tuning_trials: int | None = None
    tuning_sampler: str | None = None
    tuning_pruning_percentile: float | None = None
    tuning_pruning_warmup_steps: int | None = None
    log_model_on_validation: bool = False
    log_artifact_metrics_on_validation: bool = False


ExperimentConfigT = TypeVar("ExperimentConfigT", bound=ExperimentConfig)


class TunedExperiment(Generic[ExperimentConfigT]):
    """Class for running training experiments with hyperparameter tuning.

    Args:
        config: Config for experiment meta data.
        experiment_dir: Directory where run info is stored in.
    """

    def __init__(
        self,
        config: ExperimentConfigT,
        model_config: ModelConfig,
        experiment_dir: Path,
    ):
        self.config = config
        self.model_config = model_config
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

    @property
    def runs_dir(self) -> Path:
        """Runs directory is always relative to the experiment directory."""
        runs_dir = self.experiment_dir.joinpath("runs")
        runs_dir.mkdir(exist_ok=True)
        return runs_dir

    @abstractmethod
    def train_model(
        self,
        run_config: Mapping[str, BaseConfig],
        callbacks: list[TrainCallback],
    ) -> None:
        """Implementation of model training.

        Args:
            run_config: Configs used for training run.
            callbacks: Callbacks called during training.
        """

    @abstractmethod
    def run_config(self) -> Mapping[str, BaseConfig]:
        """Configs used for training run."""

    @abstractmethod
    def hyperparameters(self, run_config: Mapping[str, BaseConfig]) -> dict[str, Any]:
        """Hyperparameters logged for each run."""

    def _wrapped_run(
        self,
        run_name: str,
        extra_params: dict[str, Any] | None = None,
        trial: optuna.Trial | None = None,
        tuning: bool = False,
    ) -> RunResult:
        seed = Settings().seed
        set_seed(seed)
        mlflow_logger = MLFlowLogger(self.config.name)
        if extra_params is None:
            extra_params = {}
        extra_params["seed"] = seed
        original_config = self.run_config()
        overriden_config = {}
        for name, cfg in original_config.items():
            overriden_config[name] = overwrite_config(cfg, extra_params)

        mlflow_logger.start_run(run_name)

        tags = self.config.tags.copy()
        tags["stage"] = "tuning" if tuning else "training"

        mlflow_logger.set_tags(tags)

        try:
            mlflow_callback = MLFlowCallback(
                mlflow_logger=mlflow_logger,
                log_model=not tuning,
                log_model_on_validation=self.config.log_model_on_validation and not tuning,
                log_artifact_metrics_on_validation=self.config.log_artifact_metrics_on_validation
                and not tuning,
            )
            validation_callback = ValidationCallback(key_metric=self.config.key_metric)
            if tuning and trial is not None:
                optuna_callback = OptunaCallback(
                    key_metric=self.config.key_metric,
                    trial=trial,
                )
                # validation callback needs to come before mlflow callback
                callbacks = [validation_callback, mlflow_callback, optuna_callback]
            else:
                # validation callback needs to come before mlflow callback
                callbacks = [validation_callback, mlflow_callback]

            hyperparameters = self.hyperparameters(overriden_config)
            for callback in callbacks:
                callback.start_callback(hyperparameters)

            self.train_model(overriden_config, callbacks=callbacks)
        finally:
            if not tuning:
                model_artifact_uri: str | None = mlflow_logger.get_artifact_uri("model")
            else:
                model_artifact_uri = None

            run_result = RunResult(
                run_name=run_name,
                key_metric=self.config.key_metric,
                metric_value=validation_callback.best_metric,
                metrics=validation_callback.best_metrics,
                params=extra_params,
                config=overriden_config,
                model_artifact_uri=model_artifact_uri,
            )

            mlflow_logger.log_json("run", run_result.model_dump())
            mlflow_logger.end_run()

            result_path = self.runs_dir.joinpath(f"{run_result.run_name}.json")
            with open(result_path, "w") as f:
                json.dump(run_result.model_dump(), f)

        return run_result

    def run_tuning(
        self, study_name: str | None, tuning_params: TuningConfig | None = None
    ) -> RunResult:
        """Run hyperparameter tuning.

        Args:
            study_name: Name of run for which to tune.
            tuning_params: Hyperparameters to tune.
                If None, use tuning_params from experiment config.

        Returns:
            Results of best tuning run.
        """

        if tuning_params is None:
            tuning_params = self.config.tuning_params

        if study_name is None:
            study_name = get_timestamp()
        study_name = f"tuning-{study_name}"

        direction: Literal["maximize", "minimize"] = "maximize"
        if self.config.key_metric == "loss":
            direction = "minimize"

        study = run_tuning(
            trainable=partial(self._wrapped_run, tuning=True),
            tuning_params=tuning_params,
            study_name=study_name,
            runs_dir=self.runs_dir,
            num_workers=self.config.tuning_workers,
            num_trials=self.config.tuning_trials,
            tuning_sampler=self.config.tuning_sampler,
            tuning_pruning_percentile=self.config.tuning_pruning_percentile,
            tuning_pruning_warmup_steps=self.config.tuning_pruning_warmup_steps,
            direction=direction,
        )

        best_trial = study.best_trial
        results = self._get_trial_runs(study_name=study_name, trial_id=best_trial._trial_id)
        result = results[-1] if direction == "minimize" else results[0]

        result_path = self.runs_dir.joinpath(f"{study_name}-best.json")
        logger.info("Saving best result to %s", str(result_path))
        with open(result_path, "w") as f:
            json.dump(result.model_dump(), f)
        return result

    def _get_trial_runs(self, study_name: str | None, trial_id: int) -> list[RunResult]:
        runs = get_run_results(self.runs_dir, prefix="tuning")
        if study_name is not None:

            def name_filter(run: RunResult) -> bool:
                target_name = cast(str, study_name) + f"-trial-{trial_id}"
                return target_name == run.run_name

            runs = list(filter(name_filter, runs))
        else:
            runs = [run for run in runs if run.run_name.endswith(f"trial-{trial_id}")]
        if len(runs) != 1:
            raise ValueError(
                "Could not find unambigous run for trial. " f"Expected 1 run. Got {len(runs)}"
            )
        return runs

    def run_training(
        self, run_name: str | None, extra_params: dict[str, Any] | None = None
    ) -> RunResult:
        """Run model training.

        Args:
            run_name: Name of training run.
            extra_params: Parameters used to overwrite default configs.

        Returns:
            Results of training run
        """
        if run_name is None:
            run_name = get_timestamp()
        run_name = f"training-{run_name}"
        logger.info("Starting run %s", run_name)
        return self._wrapped_run(run_name, extra_params, tuning=False)

    def load_run_results(self, run_name: str | None, tuning: bool) -> list[RunResult]:
        """Load results for past runs.

        Runs are filtered by compatibility with the underlying model, run name and
        training modalities. They are ordered by the key metric (ascending if this
        is the loss, descending for all other types of metrics).

        Args:
            run_name: Name of run to get results for.
            tuning: If True, load tuning results, else load training results.

        Returns:
            List of run results.
        """
        direction: Literal["maximize", "minimize"] = "maximize"
        if self.config.key_metric == "loss":
            direction = "minimize"
        results = get_run_results(
            self.runs_dir,
            prefix="tuning" if tuning else "training",
            direction=direction,
        )

        # Filter out runs that don't match run name.
        if run_name is not None:

            def match_run_name(run_result: RunResult) -> bool:
                return cast(str, run_name) in run_result.run_name

            results = list(filter(match_run_name, results))

        # Different model configs make runs incompatible
        model_config_dict = self.model_config.model_dump()
        results = [run for run in results if run.config.get("model_config") == model_config_dict]

        return results


class TrainExperimentConfig(ExperimentConfig):
    """Config for (vanilla) pretraining or finetuning experiments."""

    train_config: TrainConfig
    model_checkpoint: Path | None = None

    # suppress model_* protected namespace warnings
    model_config = ConfigDict(protected_namespaces=())


class TrainExperiment(TunedExperiment[TrainExperimentConfig]):
    """Experiment for (vanilla) pretraining/finetuning.

    Args:
        config: Config for experiment and training.
        model_config: Config defining model to be trained.
        experiment_dir: Directory where run info is stored in.
        train_task: Task model is trained on. This can only be None if dummy_task is True.
            In that case, the TrainExperiment is only created for loading the pretrained weights
            during fine-tuning.
        dummy_task: Whether to create a dummy task that solely serves for
            loading pre-trained weights.

    Raises:
        ValueError if dummy_task is False but no train_task is given.
    """

    def __init__(
        self,
        config: TrainExperimentConfig,
        model_config: ModelConfig,
        experiment_dir: Path,
        train_task: Task | None,
        dummy_task: bool = False,
    ):
        super().__init__(config, model_config, experiment_dir)

        if dummy_task is False and train_task is None:
            raise ValueError(
                "If you do use this for loading pre-trained weights, please set dummy_task "
                "to True. Otherwise specifiy a train_task."
            )
        self.train_task = cast(Task, train_task)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train_model(
        self, run_config: Mapping[str, BaseConfig], callbacks: list[TrainCallback]
    ) -> None:
        """(Vanilla) model training.

        Args:
            run_config: Configs used for training run.
            callbacks: Callbacks called during training.
        """
        if self.train_task is None:
            raise ValueError(
                "Please specify a train_task. If this is a dummy experiment, "
                "this method cannot be used."
            )

        train_config: TrainConfig = run_config["train_config"]  # type: ignore[assignment]
        model_config: ModelConfig = run_config["model_config"]  # type: ignore[assignment]

        model = self._build_model(model_config)
        trainer = Trainer(config=train_config, callbacks=callbacks)
        trainer.train(
            model,
            task=self.train_task,
            validate_every=self.config.validate_every,
        )

        if self.train_task.test_set is not None:
            trainer.test(model, self.train_task)

    def _build_model(self, model_config: ModelConfig) -> Model:
        if self.train_task is None:
            raise ValueError(
                "Please specify a train_task. If this is a dummy experiment, "
                "this method cannot be used."
            )
        model_builder = get_model_builder(model_config)
        model = model_builder.build_classification_model(
            num_classes=self.train_task.num_classes, device=self.device
        )
        if self.config.model_checkpoint is not None:
            logger.info("Loading model weights from %s", str(self.config.model_checkpoint))
            model.load(checkpoint=self.config.model_checkpoint, load_head=False)
        return model

    def run_config(self) -> Mapping[str, BaseConfig]:
        """Configs used for training run."""
        return {
            "model_config": self.model_config,
            "train_config": self.config.train_config,
        }

    def hyperparameters(self, run_config: Mapping[str, BaseConfig]) -> dict[str, Any]:
        """Hyperparameters logged for each run."""
        train_config = cast(TrainConfig, run_config["train_config"])
        return train_config.hyperparameters()
