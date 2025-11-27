from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, cast

import torch
from eurocropsml.dataset.base import DataItem
from pydantic import ConfigDict
from torch import nn

from eurocropsmeta.utils import BaseConfig


class ModelConfig(BaseConfig):
    """Base class for model configs."""

    model_builder: str

    # suppress model_* protected namespace warnings
    model_config = ConfigDict(protected_namespaces=())


class Model(nn.Module):
    """Generic model architecture for pretraining and finetuning.

    Args:
        backbone: Model backbone to pretrain/finetune.
        head: Task-specific classification head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        device: torch.device,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.head = head
        self.to(device)
        self.device = device

    def forward(self, ipt: DataItem) -> torch.Tensor:
        out = self.backbone(ipt.data, **ipt.meta_data)
        return cast(torch.Tensor, self.head(out))

    def save(self, checkpoint: Path) -> None:
        """Save model to given checkpoint directory."""
        if not checkpoint.is_dir():
            raise ValueError("Model checkpoint must be a directory.")

        if self.encoder is not None:
            torch.save(
                self.encoder.state_dict(),
                checkpoint.joinpath("encoder.pt"),
            )
        torch.save(
            self.backbone.state_dict(),
            checkpoint.joinpath("backbone.pt"),
        )
        torch.save(
            self.head.state_dict(),
            checkpoint.joinpath("head.pt"),
        )

    def load(self, checkpoint: Path, load_head: bool = True) -> None:
        """Load weights from checkpoint.

        Args:
            checkpoint: Path to directory containing model weights.
            load_head: Whether to also load weights for the model head.
        """

        backbone_state = torch.load(checkpoint.joinpath("backbone.pt"), map_location=self.device)
        self.backbone.load_state_dict(backbone_state, strict=False)

        if self.encoder is not None:
            if checkpoint.joinpath("encoder.pt").exists():
                encoder_state = torch.load(
                    checkpoint.joinpath("encoder.pt"), map_location=self.device
                )
                self.encoder.load_state_dict(encoder_state, strict=False)
            else:
                raise FileNotFoundError("Could not load model encoder. No weights found.")

        if load_head and (head_checkpoint := checkpoint.joinpath("head.pt")).is_file():
            head_state = torch.load(head_checkpoint, map_location=self.device)
            self.head.load_state_dict(head_state)


T = TypeVar("T", bound=ModelConfig)


class ModelBuilder(ABC, Generic[T]):
    """Abstract base class for building models."""

    @abstractmethod
    def __init__(self, config: T):
        self.config = config

    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Build model backbone."""

    def build_task_encoder_backbone(self) -> nn.Module:
        """Build task encoder backbone for TIML."""
        raise NotImplementedError

    @abstractmethod
    def build_classification_head(self, num_classes: int) -> nn.Module:
        """Build classification head."""

    def build_classification_model(self, num_classes: int, device: torch.device) -> Model:
        """Build model suitable for classification with `num_classes` classes."""

        backbone = self.build_backbone()
        head = self.build_classification_head(num_classes)
        return Model(backbone=backbone, head=head, device=device)

    def reset_head(self, model: Model, num_classes: int) -> None:
        """Reset model classification head."""
        head = self.build_classification_head(num_classes)
        head.to(model.device)
        model.head = head
