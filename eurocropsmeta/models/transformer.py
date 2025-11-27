from typing import Any, Literal, cast

import torch
import torch.nn as nn
from eurocropsml.dataset.base import DataItem

from eurocropsmeta.models.base import Model, ModelBuilder, ModelConfig
from eurocropsmeta.models.encoder import TIMLTaskEncoder
from eurocropsmeta.models.location_encoding import LocationEncodingTIML
from eurocropsmeta.models.positional_encoding import PositionalEncoding


class TransformerBackbone(nn.Module):
    """Transformer Backbone Encoder with multi-headed self-attention.

    Args:
        in_channels: Number of input channels.
        d_model: Number of input features.
            Input dimension of query, key, and value's linear layers.
        encoder_layer: Instance of TransformerEncoderLayer class.
            It consists of self-attention and a feedforward network.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        location_encoding: Whether to encode location of time series input vector.
            If TIML is used with the TIMLTaskEncoder, location_encoding is turned off since
            it's applied inside the TIMLTaskEncoder.
        t: Period to use for positional encoding.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_enc_len: int,
        location_encoding: bool,
        t: int = 1000,
    ):
        self.d_model = d_model

        super().__init__()
        self._in_layernorm = nn.LayerNorm(in_channels)
        self._in_linear = torch.nn.Linear(in_channels, d_model)
        self._pos_encoding = PositionalEncoding(d_model, pos_enc_len, t)
        if location_encoding:
            self._loc_encoding = LocationEncodingTIML()

        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute model (backbone) forward pass."""
        x = x.float()
        if hasattr(self, "_loc_encoding"):
            # concatenates loc_encoding along -2
            task_info = self._loc_encoding(kwargs.get("center"))
            task_info = task_info.unsqueeze(1).expand(-1, x.size(1), -1)
            x = cast(torch.Tensor, torch.cat([x, task_info], dim=-1))
        x = self._in_layernorm(x)
        x = self._in_linear(x)
        enc_output: torch.Tensor = self._pos_encoding(x, kwargs.get("dates"))
        if (mask := kwargs.get("mask")) is not None and mask.dim() == x.dim():
            # masking of full time steps
            mask = mask.any(-1)
        encoder_out = self._encoder(enc_output, src_key_padding_mask=mask)

        return cast(torch.Tensor, encoder_out)


class TransformerConfig(ModelConfig):
    """Config for transformer model.

    Args:
        n_heads: Number of heads in the multi-head attention models.
        in_channels: Number of input channels.
        d_model: Number of input features. Input dimension of query, key, and value's linear layers.
        dim_fc: Dimensionality of query, key, and value used as input to the multi-head attention.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        location_encoding: Whether to encode location of time series input vector.
            If TIML is used with the TIMLTaskEncoder, location_encoding is turned off since
            it's applied inside the TIMLTaskEncoder.
        encoder_config: Config of TIMLTaskEncoder.
    """

    n_heads: int
    in_channels: int
    d_model: int
    dim_fc: int
    num_layers: int
    pos_enc_len: int
    location_encoding: bool = False
    encoder_config: dict[str, float | int | list[int]] | None = None

    model_builder: Literal["TransformerModelBuilder"] = "TransformerModelBuilder"


class TransformerModel(Model):
    """Model architecture for pre-training and fine-tuning with transformers.

    If there's an encoder (TIMLTaskEncoder) present, it is always applied before the backbone as
    well as before the head, as done in the original implementation. If the head is going to be
    reset, the before applied encoder does not have a big impact and it could be decided to not
    apply it before the head in case of resetting the head. However, for the sake of simplicity,
    we decided to always apply it before the head.
    """

    def forward(self, ipt: DataItem) -> torch.Tensor:
        if self.encoder is not None:
            task_embeddings = self.encoder(ipt.meta_data.get("center"))
            # first elements to be applied before backbone
            gamma, beta = task_embeddings[0][0], task_embeddings[1][0]
            ipt.data = (ipt.data * gamma) + beta
        out = self.backbone(ipt.data, **ipt.meta_data)
        if self.encoder is not None:
            # second elements to be applied before head
            gamma, beta = task_embeddings[0][1], task_embeddings[1][1]
            out = (out * gamma.unsqueeze(1)) + beta.unsqueeze(1)
        match self.head:
            case nn.Linear():
                return cast(torch.Tensor, self.head(out.mean(1)))
            case _:
                raise NotImplementedError


class TransformerModelBuilder(ModelBuilder):
    """Transformer with multi-headed self-attention.

    Args:
        config: Transformer model config.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_fc,
            batch_first=True,
        )

    def build_backbone(self) -> TransformerBackbone:
        return TransformerBackbone(
            in_channels=self.config.in_channels,
            d_model=self.config.d_model,
            encoder_layer=self._encoder_layer,
            num_layers=self.config.num_layers,
            pos_enc_len=self.config.pos_enc_len,
            location_encoding=self.config.location_encoding,
        )

    def build_task_encoder_backbone(self) -> nn.Module:
        return TIMLTaskEncoder(
            # number of linear blocks in TaskEncoder
            encoder_vector_sizes=self.config.encoder_config["encoder_vector_sizes"],
            # task length (3 since we're only using location)
            input_size=int(self.config.encoder_config["input_size"]),
            num_bands=int(self.config.in_channels),
            # number of linear classification layers to use on top of backbone
            # corresponds to number of layers in head
            num_hidden_layers=int(self.config.encoder_config["num_hidden_layers"]),
            # hidden vector size used in Transformer
            hidden_vector_size=int(self.config.d_model),
            encoder_dropout=self.config.encoder_config["encoder_dropout"],
            # always padding to 366 days since length is fixed in TaskEncoder
            num_timesteps=366,
            # number of channels for each group
            num_channels_per_group=int(self.config.encoder_config["num_channels_per_group"]),
        )

    def build_classification_head(self, num_classes: int) -> nn.Linear:
        return nn.Linear(self.config.d_model, num_classes)

    def build_classification_model(self, num_classes: int, device: torch.device) -> Model:
        if self.config.encoder_config is not None:
            encoder = self.build_task_encoder_backbone()
        else:
            encoder = None
        backbone = self.build_backbone()
        head = self.build_classification_head(num_classes)
        return TransformerModel(backbone=backbone, head=head, device=device, encoder=encoder)
