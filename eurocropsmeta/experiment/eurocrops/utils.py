import logging
import random
from pathlib import Path
from typing import Literal

import torch.nn as nn
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
    EuroCropsSplit,
)
from eurocropsml.dataset.preprocess import get_class_ids_to_names

from eurocropsmeta.dataset.eurocrops.meta import load_meta_split
from eurocropsmeta.dataset.eurocrops.pretrain import load_dataset_split
from eurocropsmeta.dataset.task import Task, TaskDataset
from eurocropsmeta.experiment.meta import MetaDatasetConfig

logger = logging.getLogger(__name__)


def load_metalearning_task(
    split_dir: Path,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    meta_dataset_config: MetaDatasetConfig,
    pad_seq_to_366: bool = False,
) -> tuple[TaskDataset, TaskDataset]:
    """Load meta-learning task from the EuroCrops dataset."""

    split = dataset_config.split

    random.seed(meta_dataset_config.task_random_seed)
    return load_meta_split(
        split_dir=split_dir,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
        meta_dataset_config=meta_dataset_config,
        split=split,
        pad_seq_to_366=pad_seq_to_366,
        loss_fn=nn.CrossEntropyLoss(),
    )


def load_non_metalearning_task(
    split_dir: Path,
    mode: Literal["pretraining", "finetuning"],
    split_config: EuroCropsSplit,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    max_samples: int | str = "all",
    pad_seq_to_366: bool = False,
) -> Task:
    """Load pretraining, meta-learning or finetuning task from the EuroCrops dataset."""

    split = dataset_config.split

    classes = (
        set(split_config.pretrain_classes[split])
        if mode == "pretraining"
        else set(split_config.finetune_classes[split])
    )
    return load_dataset_split(
        mode=mode,
        classes=classes,
        split_dir=split_dir,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
        loss_fn=nn.CrossEntropyLoss(),
        max_samples=max_samples,
        pad_seq_to_366=pad_seq_to_366,
        class_ids_to_names=get_class_ids_to_names(preprocess_config.raw_data_dir),
    )
