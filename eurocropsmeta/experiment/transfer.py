import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

from eurocropsmeta.dataset.task import Task, TaskDataset
from eurocropsmeta.experiment.base import (
    TrainExperiment,
    TrainExperimentConfig,
    TunedExperiment,
)
from eurocropsmeta.experiment.meta import MetaTrainExperiment, MetaTrainExperimentConfig
from eurocropsmeta.experiment.runs import download_run_model
from eurocropsmeta.models.base import ModelConfig

logger = logging.getLogger(__name__)


class TransferExperiment:
    """Class formalizing transfer experiment.

    Args:
        pretrain_experiment: Experiment generated pretrained weights.
        finetuning_experiments: Experiments to finetune pretrained model on.
    """

    def __init__(
        self,
        pretrain_experiment: TunedExperiment,
        finetuning_experiments: dict[str, TrainExperiment] | None,
    ):
        self.pretrain_experiment = pretrain_experiment
        self.finetuning_experiments = finetuning_experiments

    def run_pretrain_tuning(self, study_name: str | None) -> None:
        """Run pretrain tuning stage."""
        logger.info("Running hyperparameter tuning for pretraining experiment.")
        self.pretrain_experiment.run_tuning(study_name=study_name)

    def run_pretraining(self, run_name: str | None) -> None:
        """Run pretraining stage."""
        logger.info("Running training for pretraining experiment.")
        extra_params = self._load_tuning_params(
            experiment=self.pretrain_experiment, study_name=run_name
        )

        self.pretrain_experiment.run_training(run_name=run_name, extra_params=extra_params)

    def run_finetune_tuning(
        self,
        task_name: str,
        study_name: str | None,
        pretrain_run_name: str | None = "auto",
    ) -> None:
        """Run finetuning tuning stage for given task.

        If `pretrain_run_name` is given, this training run is used for pretrained weights.
        If it is 'auto', then the best available run is used.
        Otherwise, the weights are randomly initialized.
        """
        logger.info("Running hyperparameter tuning for finetuning task %s.", task_name)
        if self.finetuning_experiments is not None:
            experiment = self._get_finetunig_experiment(task_name, pretrain_run_name)
        else:
            raise ValueError(
                "Please set mode to 'finetune' when building the experiment. "
                "Otherwise the finetune experiment is not created."
            )

        experiment.run_tuning(study_name=study_name)

    def run_finetuning(
        self,
        task_name: str,
        run_name: str | None,
        pretrain_run_name: str | None = "auto",
    ) -> None:
        """Run finetuning stage for given task.

        If `pretrain_run_name` is given, this training run is used for pretrained weights.
        If it is 'auto', then the best available run is used.
        Otherwise, the weights are randomly initialized.
        """
        logger.info("Running training for finetuning task %s.", task_name)
        if self.finetuning_experiments is not None:
            experiment = self._get_finetunig_experiment(task_name, pretrain_run_name)
        else:
            raise ValueError(
                "Please set mode to 'finetune' when building the experiment. "
                "Otherwise the finetune experiment is not created."
            )
        extra_params = self._load_tuning_params(experiment=experiment, study_name=run_name)
        experiment.run_training(run_name=run_name, extra_params=extra_params)

    def _resolve_pretrain_run(self, pretrain_run_name: str) -> tuple[str, Path]:
        if pretrain_run_name == "auto":
            pretrain_results = self.pretrain_experiment.load_run_results(
                run_name=None, tuning=False
            )
            if not pretrain_results:
                raise ValueError(
                    "No suitable pretrained results found in "
                    + str(self.pretrain_experiment.experiment_dir)
                )
            pretrain_result = pretrain_results[0]
            pretrain_run_name = pretrain_result.run_name
        logger.info("Using %s as pretrained checkpoint.", pretrain_run_name)
        model_checkpoint = self.pretrain_experiment.experiment_dir.joinpath(
            pretrain_run_name, "model"
        )
        if not model_checkpoint.is_dir():
            logger.info(
                "Trying to pull model weights for run id %s",
                pretrain_run_name,
            )
            download_run_model(
                experiment_name=self.pretrain_experiment.config.name,
                run_name=pretrain_run_name,
                runs_dir=self.pretrain_experiment.experiment_dir,
            )
        logger.info("Using pretrained weights from %s", str(model_checkpoint))

        return pretrain_run_name, model_checkpoint

    def _get_finetunig_experiment(
        self, task_name: str, pretrain_run_name: str | None = None
    ) -> TrainExperiment:
        self.finetuning_experiments = cast(dict[str, TrainExperiment], self.finetuning_experiments)
        if task_name not in self.finetuning_experiments:
            raise ValueError(
                f"Task {task_name} not defined."
                f"Please use one of the following: {self.finetuning_experiments.keys()}"
            )

        experiment = self.finetuning_experiments[task_name]

        if pretrain_run_name is not None:
            pretrain_run_name, model_checkpoint = self._resolve_pretrain_run(pretrain_run_name)
            logger.info("Using pretrained weights from %s", str(model_checkpoint))
            experiment.config.model_checkpoint = model_checkpoint
            experiment.config.tags["pretrain_run_name"] = str(pretrain_run_name)
            experiment.config.tags["pretrain_experiment_name"] = str(
                self.pretrain_experiment.config.name
            )
        else:
            logger.info("Using randomly initialized weights.")
            experiment.config.model_checkpoint = None
            experiment.config.tags["pretrain_run_name"] = "None"
            experiment.config.tags["pretrain_experiment_name"] = "None"

        experiment.experiment_dir = experiment.experiment_dir.joinpath(
            experiment.config.tags["pretrain_experiment_name"]
            + " -- "
            + experiment.config.tags["pretrain_run_name"]
        )
        experiment.experiment_dir.mkdir(exist_ok=True)

        return experiment

    @staticmethod
    def _load_tuning_params(experiment: TunedExperiment, study_name: str | None) -> dict[str, Any]:
        tuning_results = experiment.load_run_results(run_name=study_name, tuning=True)
        if not tuning_results:
            logger.warning("No results from hyperparameter tuning found. Using default params.")
            return {}

        logger.info(
            "Found %s valid results from hyperparameter tuning.",
            len(tuning_results),
        )
        tuning_run = tuning_results[0]  # results are sorted, we want the first
        logger.info(
            "Using params from run %s (params: %s)",
            tuning_run.run_name,
            tuning_run.params,
        )

        return tuning_run.params


def build_pretrain_experiment(
    pretrain_experiment_config: TrainExperimentConfig,
    pretrain_task: Task | None,
    finetuning_tasks: dict[str, tuple[Task, TrainExperimentConfig]] | None,
    model_config: ModelConfig,
    experiment_dir: Path,
) -> TransferExperiment:
    """Build transfer experiment with regular pretraining.

    Args:
        pretrain_experiment_config: Experiment config of the pretraining experiment.
        pretrain_task: Pretraining task to train on.
        finetuning_tasks: Mapping between finetuning task name and task with experiment config.
        model_config: Model config specifying model to be trained.
        experiment_dir: Directory where run results are stored in.

    Returns:
        Specified transfer experiment.
    """
    logger.info("Building regular pretraining transfer experiment.")

    pretrain_experiment = TrainExperiment(
        config=pretrain_experiment_config,
        experiment_dir=experiment_dir.joinpath(pretrain_experiment_config.name),
        train_task=pretrain_task,
        model_config=model_config,
        dummy_task=pretrain_task is None,
    )
    if finetuning_tasks is not None:
        finetuning_experiments = _build_finetuning_experiments(
            finetuning_tasks=finetuning_tasks,
            model_config=model_config,
            base_dir=experiment_dir,
        )
    else:
        finetuning_experiments = None
    return TransferExperiment(
        pretrain_experiment=pretrain_experiment,
        finetuning_experiments=finetuning_experiments,
    )


def build_metalearn_experiment(
    meta_experiment_config: MetaTrainExperimentConfig,
    train_set: TaskDataset | None,
    val_set: TaskDataset | None,
    finetuning_tasks: dict[str, tuple[Task, TrainExperimentConfig]] | None,
    model_config: ModelConfig,
    experiment_dir: Path,
) -> TransferExperiment:
    """Build transfer experiment with meta-pretraining.

    Args:
        meta_experiment_config: Experiment config of the meta-training experiment.
        train_set: Task dataset used for (meta-) pretraining.
        val_set: Task dataset used for (meta-) validation.
        finetuning_tasks: Mapping between finetuning task name and task with experiment config.
        model_config: Model config specifying model to be trained.
        experiment_dir: Directory where run results are stored in.

    Returns:
        Specified transfer experiment.
    """

    logger.info("Building metalearning transfer experiment.")

    meta_experiment = MetaTrainExperiment(
        config=meta_experiment_config,
        experiment_dir=experiment_dir.joinpath(meta_experiment_config.name),
        model_config=model_config,
        train_set=train_set,
        val_set=val_set,
        dummy_task=(train_set is None and val_set is None),
    )
    if finetuning_tasks is not None:
        finetuning_experiments = _build_finetuning_experiments(
            finetuning_tasks=finetuning_tasks,
            model_config=model_config,
            base_dir=experiment_dir,
        )
    else:
        finetuning_experiments = None
    return TransferExperiment(
        pretrain_experiment=meta_experiment,
        finetuning_experiments=finetuning_experiments,
    )


def _build_finetuning_experiments(
    finetuning_tasks: dict[str, tuple[Task, TrainExperimentConfig]],
    model_config: ModelConfig,
    base_dir: Path,
) -> dict[str, TrainExperiment]:
    logger.info("Building finetuning experiments.")
    experiments = {}
    for name, (task, config) in finetuning_tasks.items():
        config = config.copy(deep=True)
        config.tags["Task"] = name
        experiment = TrainExperiment(
            config=config,
            experiment_dir=base_dir.joinpath(config.name).joinpath(name),
            train_task=task,
            model_config=model_config,
        )

        experiments[name] = experiment

    return experiments


ExperimentConfigT = TypeVar("ExperimentConfigT")


class TransferExperimentBuilder(Generic[ExperimentConfigT]):
    """Base class for building transfer experiments."""

    def __init__(self, config: ExperimentConfigT):
        self.config = config

    @abstractmethod
    def build_experiment(self, mode: Literal["pretrain", "finetune"]) -> TransferExperiment:
        """Build transfer experiment."""
