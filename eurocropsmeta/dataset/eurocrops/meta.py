import json
import logging
from collections import defaultdict
from functools import reduce
from itertools import islice
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
from tqdm import tqdm

from eurocropsmeta.dataset.task import (
    ClassificationTaskDataset,
    FixedTaskDataset,
    WrapClassificationTaskDataset,
)
from eurocropsmeta.experiment.meta import MetaDatasetConfig

logger = logging.getLogger(__name__)


def _build_meta_class_dataset(
    files_dict: dict[str, dict[str, list[Path]]],
    dataset_config: EuroCropsDatasetConfig,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    meta_dataset_config: MetaDatasetConfig,
    sample: bool,
    loss_fn: nn.Module,
    pad_seq_to_366: bool,
) -> WrapClassificationTaskDataset:
    encode = {int(c): i for i, c in enumerate(files_dict.keys())}

    datasets = {}
    file_list = reduce(
        add, [reduce(add, region_dict.values()) for region_dict in files_dict.values()]
    )

    mmap_store = MMapStore(map(Path, file_list))
    for c, files in tqdm(files_dict.items()):
        datasets[encode[int(c)]] = TransformDataset(
            EuroCropsDataset(
                file_dict=files,
                encode=encode,
                mmap_store=mmap_store,
                config=dataset_config,
                preprocess_config=preprocess_config,
                pad_seq_to_366=pad_seq_to_366,
            ),
            collate_fn=custom_collate_fn,
        )

    return ClassificationTaskDataset(
        datasets=datasets,
        num_classes=meta_dataset_config.num_classes,
        train_samples_per_class=meta_dataset_config.train_samples_per_class,
        test_samples_per_class=meta_dataset_config.test_samples_per_class,
        sample=sample,
        loss_fn=loss_fn,
        metrics_list=dataset_config.metrics,
    )


def _build_meta_region_dataset(
    region_files: dict[str, dict[str, dict[str, list[Path]]]],
    region_encoding: dict[str, int],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    meta_dataset_config: MetaDatasetConfig,
    sample: bool,
    loss_fn: nn.Module,
    pad_seq_to_366: bool,
) -> WrapClassificationTaskDataset:

    def _build_single_region_dict(region: str) -> dict[int, TransformDataset]:

        if region in region_files:
            class_list = reduce(
                add,
                [reduce(add, class_dict.values()) for class_dict in region_files[region].values()],
            )

            region_mmap = MMapStore(map(Path, class_list))

            return {
                i: TransformDataset(
                    EuroCropsDataset(
                        file_dict=file_dict,
                        encode={int(files_id): i},
                        mmap_store=region_mmap,
                        config=dataset_config,
                        preprocess_config=preprocess_config,
                        pad_seq_to_366=pad_seq_to_366,
                    ),
                    collate_fn=custom_collate_fn,
                )
                for i, (files_id, file_dict) in enumerate(region_files[region].items())
            }
        else:  # this region is missing in the files, so we skip it
            return {}

    data_dict = {
        region_id: _build_single_region_dict(region)
        for region, region_id in region_encoding.items()
    }

    return WrapClassificationTaskDataset(
        datasets=data_dict,
        num_classes=meta_dataset_config.num_classes,
        train_samples_per_class=meta_dataset_config.train_samples_per_class,
        test_samples_per_class=meta_dataset_config.test_samples_per_class,
        sample=sample,
        loss_fn=loss_fn,
        metrics_list=dataset_config.metrics,
    )


def load_meta_split(
    split: Literal["class", "regionclass", "region"],
    split_dir: Path,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    meta_dataset_config: MetaDatasetConfig,
    loss_fn: nn.Module,
    pad_seq_to_366: bool,
) -> tuple[WrapClassificationTaskDataset, FixedTaskDataset]:
    """Load EuroCrops training dataset for meta-learning, split by classes.

    Args:
        split: Kind of data split to apply.
        split_dir: Directory where split file containing the data splits is saved
        preprocess_config: Config model of preprocessed data.
        dataset_config: Config model of dataset to be loaded.
        meta_dataset_config: Config model of meta dataset to be loaded.
        loss_fn: Loss function used to calculate the model's loss.
        pad_seq_to_366: If sequence should be padded to 366 days
            This is only used for TIML with an encoder.

    Returns:
        Meta-train and meta-val datasets.

    Raises:
        FileNotFoundError: If split-file to load data split does not exist.
    """

    split_file = split_dir.joinpath("meta", f"{split}_split.json")

    if split_file.exists():
        with open(split_file) as outfile:
            data_split = json.load(outfile)

    else:
        raise FileNotFoundError(
            str(split_file) + " does not exist. Please first build the dataset split."
        )

    meta_train = data_split["train"]
    meta_val = data_split["val"]

    satellites = dataset_config.data_sources
    satellites.sort()

    if split == "class":
        train: dict = defaultdict(lambda: defaultdict(list))
        [
            train[c][s].append(
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
            )
            for c, file_list in meta_train.items()
            for s in satellites
            for file in file_list
        ]
        logger.debug("Computing class meta-train tasks.")
        meta_train_set = _build_meta_class_dataset(
            files_dict=train,
            dataset_config=dataset_config,
            meta_dataset_config=meta_dataset_config,
            preprocess_config=preprocess_config,
            sample=True,
            loss_fn=loss_fn,
            pad_seq_to_366=pad_seq_to_366,
        )

        val: dict = defaultdict(lambda: defaultdict(list))
        [
            val[c][s].append(
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
            )
            for c, file_list in meta_val.items()
            for s in satellites
            for file in file_list
        ]
        logger.debug("Computing class meta-val tasks.")
        meta_val_set = _build_meta_class_dataset(
            files_dict=val,
            dataset_config=dataset_config,
            meta_dataset_config=meta_dataset_config,
            preprocess_config=preprocess_config,
            sample=True,
            loss_fn=loss_fn,
            pad_seq_to_366=pad_seq_to_366,
        )
    else:
        encoding = {
            region: i
            for i, region in enumerate(set(list(meta_train.keys()) + list(meta_val.keys())))
        }

        train = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        [
            train[region][c][s].append(
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
            )
            for region in meta_train
            for c, file_list in meta_train[region].items()
            for s in satellites
            for file in file_list
        ]
        logger.info(f"Computing {split} meta-train tasks.")
        meta_train_set = _build_meta_region_dataset(
            region_files=train,
            region_encoding=encoding,
            preprocess_config=preprocess_config,
            dataset_config=dataset_config,
            meta_dataset_config=meta_dataset_config,
            sample=True,
            loss_fn=loss_fn,
            pad_seq_to_366=pad_seq_to_366,
        )

        logger.info(f"Computing {split} meta-val tasks.")
        val = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        [
            val[region][c][s].append(
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
            )
            for region in meta_val
            for c, file_list in meta_val[region].items()
            for s in satellites
            for file in file_list
        ]
        meta_val_set = _build_meta_region_dataset(
            region_files=val,
            region_encoding=encoding,
            preprocess_config=preprocess_config,
            dataset_config=dataset_config,
            meta_dataset_config=meta_dataset_config,
            sample=True,
            loss_fn=loss_fn,
            pad_seq_to_366=pad_seq_to_366,
        )

    meta_val_tasks = list(tqdm(islice(meta_val_set, meta_dataset_config.num_test_tasks)))
    if len(meta_val_tasks) < meta_dataset_config.num_test_tasks:
        logger.warning(
            f"Trying to obtain {meta_dataset_config.num_test_tasks} sample(s)"
            f"from a dataset that contains only {len(meta_val_tasks)} sample(s)."
            f"This might result in an unintended behavior."
        )

    return (
        meta_train_set,
        FixedTaskDataset(meta_val_tasks),
    )
