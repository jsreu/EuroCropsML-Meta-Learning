from typing import Literal

import eurocropsml.settings
from eurocropsml.dataset.splits import get_split_dir

from eurocropsmeta.experiment.eurocrops.config import EuroCropsTransferConfig
from eurocropsmeta.experiment.eurocrops.utils import (
    load_metalearning_task,
    load_non_metalearning_task,
)
from eurocropsmeta.experiment.meta import MetaTrainExperimentConfig
from eurocropsmeta.experiment.transfer import (
    TransferExperiment,
    TransferExperimentBuilder,
    build_metalearn_experiment,
    build_pretrain_experiment,
)
from eurocropsmeta.settings import Settings


class EuroCropsExperimentBuilder(TransferExperimentBuilder[EuroCropsTransferConfig]):
    """Class for building EuroCrops related experiments."""

    def __init__(self, config: EuroCropsTransferConfig):
        super().__init__(config)
        self.dataset_config = config.eurocrops_dataset

    def build_experiment(self, mode: Literal["pretrain", "finetune"]) -> TransferExperiment:
        """Build transfer experiment."""
        model_channels = self.config.model.in_channels
        if self.config.model.location_encoding is True:
            # for TIML without encoder (location_encoding=True), 3 additional channels are added
            model_channels -= 3
        if model_channels != self.dataset_config.total_num_channels:
            raise AssertionError(
                "The number of channels in the model config "
                f"({self.config.model.in_channels}) does not match the number of actual channels "
                f"from the dataset config ({self.dataset_config.total_num_channels}). Please "
                "adjust the number of channels in the model config and make sure you are using the"
                " correct set of data sources as well as are not accidentally removing any S2 "
                "bands you were not planning to remove."
            )

        experiment_dir = Settings().experiment_dir
        experiment_dir.mkdir(exist_ok=True, parents=True)

        data_dir = eurocropsml.settings.Settings().data_dir

        split_dir = get_split_dir(data_dir, self.config.split.base_name)

        if mode == "finetune":
            finetuning_tasks = {
                f"eurocrops_finetuning_maxsamples_{num}": (
                    load_non_metalearning_task(
                        split_dir,
                        mode="finetuning",
                        split_config=self.config.split,
                        preprocess_config=self.config.preprocess,
                        dataset_config=self.dataset_config,
                        max_samples=num,
                        pad_seq_to_366=getattr(self.config.model, "encoder_config", None)
                        is not None,
                    ),
                    self.config.finetune,
                )
                for num in self.dataset_config.max_samples
            }

            if isinstance(self.config.pretrain, MetaTrainExperimentConfig):
                return build_metalearn_experiment(
                    meta_experiment_config=self.config.pretrain,
                    train_set=None,
                    val_set=None,
                    finetuning_tasks=finetuning_tasks,
                    model_config=self.config.model,
                    experiment_dir=experiment_dir,
                )
            else:
                return build_pretrain_experiment(
                    pretrain_experiment_config=self.config.pretrain,
                    pretrain_task=None,
                    finetuning_tasks=finetuning_tasks,
                    model_config=self.config.model,
                    experiment_dir=experiment_dir,
                )

        elif mode == "pretrain":
            if isinstance(self.config.pretrain, MetaTrainExperimentConfig):
                metatrain_set, metaval_set = load_metalearning_task(
                    split_dir,
                    preprocess_config=self.config.preprocess,
                    dataset_config=self.dataset_config,
                    meta_dataset_config=self.config.pretrain.meta_dataset_config,
                    pad_seq_to_366=getattr(self.config.model, "encoder_config", None) is not None,
                )

                return build_metalearn_experiment(
                    meta_experiment_config=self.config.pretrain,
                    train_set=metatrain_set,
                    val_set=metaval_set,
                    finetuning_tasks=None,
                    model_config=self.config.model,
                    experiment_dir=experiment_dir,
                )
            else:
                pretrain_task = load_non_metalearning_task(
                    split_dir,
                    mode="pretraining",
                    split_config=self.config.split,
                    preprocess_config=self.config.preprocess,
                    dataset_config=self.dataset_config,
                )
                return build_pretrain_experiment(
                    pretrain_experiment_config=self.config.pretrain,
                    pretrain_task=pretrain_task,
                    finetuning_tasks=None,
                    model_config=self.config.model,
                    experiment_dir=experiment_dir,
                )

        else:
            raise ValueError(f"{mode} not implemented. Choose either 'pretrain' or 'finetune'.")
