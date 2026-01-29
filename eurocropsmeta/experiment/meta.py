from pathlib import Path
from typing import Any, Mapping, cast

from eurocropsmeta.algorithms.alfa import ALFAConfig
from eurocropsmeta.algorithms.anil import ANILConfig
from eurocropsmeta.algorithms.base import MetaLearnAlgorithmConfig
from eurocropsmeta.algorithms.maml import MAMLConfig
from eurocropsmeta.algorithms.timl import TIMLConfig
from eurocropsmeta.dataset.task import TaskDataset
from eurocropsmeta.experiment.base import ExperimentConfig, TunedExperiment
from eurocropsmeta.experiment.runs import RunResult
from eurocropsmeta.experiment.utils import get_meta_learner, get_model_builder
from eurocropsmeta.models.base import ModelConfig
from eurocropsmeta.train.callback import TrainCallback
from eurocropsmeta.train.meta_train import MetaTrainer
from eurocropsmeta.utils import BaseConfig


class MetaDatasetConfig(BaseConfig):
    """Config for meta-training dataset."""

    num_classes: int = 3
    train_samples_per_class: int = 1
    test_samples_per_class: int = 1
    num_test_tasks: int = 10
    task_random_seed: int = 42


class MetaTrainExperimentConfig(ExperimentConfig):
    """Config for meta-training experiments."""

    meta_config: MAMLConfig | ANILConfig | ALFAConfig | TIMLConfig
    meta_dataset_config: MetaDatasetConfig


class MetaTrainExperiment(TunedExperiment[MetaTrainExperimentConfig]):
    """Experiment for (vanilla) pretraining/finetuning.

    Args:
        config: Config for experiment and meta-training config.
        model_config: Config defining model to be trained.
        experiment_dir: Directory where run info is stored in.
        train_set: Task dataset used for meta-training.
            This can only be None if dummy_task is True.
            In that case, the TrainExperiment is only created for loading the pretrained weights
            during fine-tuning.
        val_set: Task dataset used for meta-validation.
            This can only be None if dummy_task is True.
            In that case, the TrainExperiment is only created for loading the pretrained weights
            during fine-tuning.
        dummy_task: Whether to create a dummy task that solely serves for
            loading meta-learned pretraining weights.
    """

    def __init__(
        self,
        config: MetaTrainExperimentConfig,
        model_config: ModelConfig,
        experiment_dir: Path,
        train_set: TaskDataset | None,
        val_set: TaskDataset | None,
        dummy_task: bool = False,
    ):
        super().__init__(
            config,
            model_config,
            experiment_dir,
        )
        if dummy_task is False and None in [train_set, val_set]:
            raise ValueError(
                "If you do use this for loading pre-trained weights, please set dummy_task "
                "to True. Otherwise specifiy a train_set and val_set."
            )
        self.train_set = cast(TaskDataset, train_set)
        self.val_set = cast(TaskDataset, val_set)

    def train_model(
        self, run_config: Mapping[str, BaseConfig], callbacks: list[TrainCallback]
    ) -> None:
        """Model training using meta-learning.

        Args:
            run_config: Configs used for training run.
            callbacks: Callbacks called during training.
        """
        if None in [self.train_set, self.val_set]:
            raise ValueError(
                "Please specify a train_set and val_set. If this is a dummy experiment, "
                "this method cannot be used."
            )
        meta_config = cast(MetaLearnAlgorithmConfig, run_config["meta_config"])
        meta_dataset_config = self.config.meta_dataset_config
        model_config = cast(ModelConfig, run_config["model_config"])
        model_builder = get_model_builder(model_config)
        meta_learner = get_meta_learner(meta_config, model_builder)
        meta_learner.config.num_classes = meta_config.num_classes
        if meta_config.prefetch_data:
            prefetch_task_data = (
                meta_dataset_config.num_classes * meta_dataset_config.train_samples_per_class
            )
        else:
            prefetch_task_data = None
        meta_trainer = MetaTrainer(
            meta_learner, prefetch_task_data=prefetch_task_data, callbacks=callbacks
        )

        if None not in [self.train_set, self.val_set]:
            meta_trainer.train(
                self.train_set,
                self.val_set,
                validate_every=self.config.validate_every,
            )

    def run_config(self) -> Mapping[str, BaseConfig]:
        """Configs used for training run."""
        return {
            "model_config": self.model_config,
            "meta_config": self.config.meta_config,
            "meta_dataset_config": self.config.meta_dataset_config,
        }

    def hyperparameters(self, run_config: Mapping[str, BaseConfig]) -> dict[str, Any]:
        """Hyperparameters logged for each run."""
        meta_config = cast(MetaLearnAlgorithmConfig, run_config["meta_config"])
        return meta_config.hyperparameters() | self.config.meta_dataset_config.model_dump()

    def load_run_results(self, run_name: str | None, tuning: bool) -> list[RunResult]:
        """Load results for past training runs.

        Runs are filtered by compatibility with the underlying model, run name and
        training modalities. They are ordered by the key metric (ascending if this
        is the loss, descending for all other types of metrics).

        Args:
            run_name: Name of run to get results for.
                If None, results are not filtered by name.
            tuning: If True, load tuning results, else load training results.

        Returns:
            List of run results.
        """
        results = super().load_run_results(run_name=run_name, tuning=tuning)

        if tuning:
            # different meta algorithm and dataset configs make runs incompatible

            def compatible(run: RunResult) -> bool:
                # number of classes match
                class_check = self.config.meta_dataset_config.num_classes == cast(
                    int, run.config["meta_dataset_config"]["num_classes"]
                )
                # training samples per class match
                samples_check = self.config.meta_dataset_config.train_samples_per_class == cast(
                    int,
                    run.config["meta_dataset_config"]["train_samples_per_class"],
                )
                # task adaption steps match
                steps_check = self.config.meta_config.train_adaption_steps == cast(
                    int, run.config["meta_config"]["train_adaption_steps"]
                )
                return all([class_check, samples_check, steps_check])

            results = [run for run in results if compatible(run)]

        return results
