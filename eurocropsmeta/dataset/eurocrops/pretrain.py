import json
import logging
from functools import reduce
from operator import add
from pathlib import Path
from typing import Literal

import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset, custom_collate_fn
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
)
from eurocropsml.dataset.dataset import EuroCropsDataset
from eurocropsml.dataset.utils import MMapStore

from eurocropsmeta.dataset.task import Task
from eurocropsmeta.train.utils import get_metrics

logger = logging.getLogger(__name__)


def load_dataset_split(
    mode: Literal["pretraining", "finetuning"],
    classes: set,
    split_dir: Path,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    loss_fn: nn.Module,
    pad_seq_to_366: bool,
    max_samples: int | str,
    class_ids_to_names: dict[str, str] | None,
) -> Task:
    """Load EuroCrops data.

    Args:
        mode: Whether to load pretrain or finetuning dataset.
        classes: The classes of the requested dataset split.
        split_dir: Directory where split is loaded from.
        preprocess_config: Config model of preprocessed data.
        dataset_config: Config model of dataset to be loaded.
        loss_fn: Loss function used to calculate the model's loss.
        max_samples: Maximum number of samples per class within finetuning dataset.
        pad_seq_to_366: If sequence should be padded to 366 days
            This is only used for fine-tuning TIML with an encoder.
        class_ids_to_names: Optional mapping from class identifiers to readable class names.

    Returns:
        Task containing train, validation, and optionally test dataset.

    Raises:
        FileNotFoundError: If dataset split file is not found.
        FileNotFoundError: If train and/or validation set do not exist.
        FileNotFoundError: If finetuning mode and test set does not exist.
    """

    class_list = list(classes)  # ensure matching ordering between names and encoding
    if class_ids_to_names is not None:  # use readable class names if available
        class_names = [class_ids_to_names[str(c)] for c in class_list]
    else:  # use class identifiers as names otherwise
        class_names = [str(c) for c in class_list]
    encoding = {int(c): i for i, c in enumerate(class_list)}

    if mode == "finetuning":
        split_file = split_dir.joinpath(
            "finetune", f"{dataset_config.split}_split_{max_samples}.json"
        )
    else:
        split_file = split_dir.joinpath("pretrain", f"{dataset_config.split}_split.json")
    if split_file.exists():
        with open(split_file) as outfile:
            data_split = json.load(outfile)

    else:
        raise FileNotFoundError(
            str(split_file) + " does not exist. Please first build the dataset split."
        )

    satellites = dataset_config.data_sources
    satellites.sort()

    data_satellite_split: dict[str, dict[str, list[Path]]] = {
        key: {s: [] for s in satellites} for key in data_split
    }
    data_satellite_split = {
        key: {
            s: [
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
                for file in file_list
            ]
            for s in satellites
        }
        for key, file_list in data_split.items()
    }

    try:
        train = data_satellite_split["train"]
        train_list = reduce(add, train.values())
        val = data_satellite_split["val"]
        val_list = reduce(add, val.values())
        if mode == "finetuning":
            test = data_satellite_split["test"]
            test_list = [item for sublist in test.values() for item in sublist]
        else:
            test = None
            test_list = None
    except KeyError as err:
        raise FileNotFoundError() from err

    logger.info(f"Computing {mode} task.")
    mmap_store = MMapStore(train_list + val_list + (test_list if test_list is not None else []))
    metrics = get_metrics(
        dataset_config.metrics,
        num_classes=len(class_names),
        class_names=class_names,
    )
    task = Task(
        task_id="eurocrops",
        train_set=TransformDataset(
            EuroCropsDataset(
                train,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                pad_seq_to_366=pad_seq_to_366,
                preprocess_config=preprocess_config,
            ),
            collate_fn=custom_collate_fn,
        ),
        val_set=TransformDataset(
            EuroCropsDataset(
                val,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                pad_seq_to_366=pad_seq_to_366,
                preprocess_config=preprocess_config,
            ),
            collate_fn=custom_collate_fn,
        ),
        test_set=(
            TransformDataset(
                EuroCropsDataset(
                    test,
                    encode=encoding,
                    mmap_store=mmap_store,
                    config=dataset_config,
                    pad_seq_to_366=pad_seq_to_366,
                    preprocess_config=preprocess_config,
                ),
                collate_fn=custom_collate_fn,
            )
            if test
            else None
        ),
        num_classes=len(encoding.keys()),
        loss_fn=loss_fn,
        metrics=metrics,
    )

    return task
