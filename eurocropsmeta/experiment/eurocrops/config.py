import logging
from typing import Any

from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
    EuroCropsSplit,
)
from pydantic import BaseModel

from eurocropsmeta.experiment.base import TrainExperimentConfig
from eurocropsmeta.experiment.meta import MetaTrainExperimentConfig
from eurocropsmeta.models.transformer import TransformerConfig

logger = logging.getLogger(__name__)


class EuroCropsTransferConfig(BaseModel):
    """Configuration for the EuroCrops transfer experiment.

    Args:
        base_name: Name under which experiments are stored.
        model: Configuration for a (pre)-trained model.
        pretrain: Experiment configuration for pretraining.
        finetune: Experiment configuration for finetuning.
        eurocrops_dataset: Configuration for EuroCrops dataset.
        split: Configuration for EuroCrops splits.
    """

    base_name: str
    model: TransformerConfig
    pretrain: TrainExperimentConfig | MetaTrainExperimentConfig
    finetune: TrainExperimentConfig

    eurocrops_dataset: EuroCropsDatasetConfig

    split: EuroCropsSplit
    preprocess: EuroCropsDatasetPreprocessConfig

    def __init__(self, **data: Any):
        super().__init__(**data)
