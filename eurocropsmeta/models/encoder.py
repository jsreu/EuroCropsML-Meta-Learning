from typing import List, Optional, Tuple, cast

import torch
from torch import nn

from eurocropsmeta.models.location_encoding import LocationEncodingTIML


class TIMLTaskEncoder(nn.Module):
    """Encoder module for encoding task information for TIML algorithm.

    Args:
        input_size: Size of the task info.
        encoder_vector_sizes: Number of linear blocks in TaskEncoder.
        num_bands: Number of bands.
        num_hidden_layers: Number of linear classification layers to use.
        hidden_vector_size: Number of input features for backbone.
            Input dimension of query, key, and value's linear layers.
        encoder_dropout: Percentage of encoder dropout.
        num_timesteps: Sequence length.
        num_channels_per_group: Number of channels per group for each
            group normalization.
    """

    def __init__(
        self,
        input_size: int,
        encoder_vector_sizes: List[int],
        num_bands: int,
        num_hidden_layers: int,
        hidden_vector_size: int,
        encoder_dropout: float,
        num_timesteps: int,
        num_channels_per_group: int | List[int] = 16,
    ) -> None:
        super().__init__()

        self._loc_encoding = LocationEncodingTIML()

        if isinstance(num_channels_per_group, int):
            num_channels_per_group = [num_channels_per_group for _ in encoder_vector_sizes]

        for idx, vector_size in enumerate(encoder_vector_sizes):
            assert vector_size % num_channels_per_group[idx] == 0

        encoder_layers: List[nn.Module] = []  # linear blocks
        # each linear block consists of:
        # - linear layer
        # - GeLU activation function
        # - group normalization
        # - dropout
        for i in range(len(encoder_vector_sizes)):
            encoder_layers.append(
                nn.Linear(
                    in_features=input_size if i == 0 else encoder_vector_sizes[i - 1],
                    out_features=encoder_vector_sizes[i],
                )
            )
            encoder_layers.append(nn.GELU())
            encoder_layers.append(
                nn.GroupNorm(
                    num_channels=encoder_vector_sizes[i],
                    num_groups=encoder_vector_sizes[i] // num_channels_per_group[i],
                )
            )
            encoder_layers.append(nn.Dropout(p=encoder_dropout))

        self.initial_encoder = nn.Sequential(*encoder_layers)  # TaskEncoder

        # encoding of task information
        self.gamma_layers: nn.ModuleDict = nn.ModuleDict()
        self.beta_layers: nn.ModuleDict = nn.ModuleDict()
        # apply once before backbone and once before every linear classification layer
        for i in range(num_hidden_layers + 1):
            # these will want outputs of shape [hidden_vector_size, hidden_vector_size]
            gamma_layer: Linear3d | nn.Sequential
            beta_layer: Linear3d | nn.Sequential
            if i == 0:
                # the nonlinearity is captured in the linear3d layer
                gamma_layer = Linear3d(
                    in_features=encoder_vector_sizes[-1],
                    out_shape=(num_timesteps, num_bands),
                    sum_from=1,
                )
                beta_layer = Linear3d(
                    in_features=encoder_vector_sizes[-1],
                    out_shape=(num_timesteps, num_bands),
                    sum_from=0,
                )
            else:
                gamma_layer = cast(
                    nn.Sequential,
                    nn.Sequential(
                        nn.Linear(
                            in_features=encoder_vector_sizes[-1],
                            out_features=hidden_vector_size,
                        ),
                        nn.GELU(),
                    ),
                )
                beta_layer = cast(
                    nn.Sequential,
                    nn.Sequential(
                        nn.Linear(
                            in_features=encoder_vector_sizes[-1],
                            out_features=hidden_vector_size,
                        ),
                        nn.GELU(),
                    ),
                )

            self.gamma_layers[f"task_embedding_{i}_gamma"] = gamma_layer
            self.beta_layers[f"task_embedding_{i}_beta"] = beta_layer

        self.dropout = nn.Dropout(p=encoder_dropout)

    def forward(self, location: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        task_info = self._loc_encoding(location)  # task_info
        task_info = self.initial_encoder(task_info)

        gamma_outputs: List[torch.Tensor] = []
        beta_outputs: List[torch.Tensor] = []
        for gamma_layer in self.gamma_layers.values():
            # g_layer = cast(nn.Module, self.__getattr__(layer_name))
            gamma_outputs.append(self.dropout(gamma_layer(task_info)))
        for beta_layer in self.beta_layers.values():
            # b_layer = cast(nn.Module, self.__getattr__(layer_name))
            beta_outputs.append(self.dropout(beta_layer(task_info)))
        return (gamma_outputs, beta_outputs)


class Linear3d(nn.Module):
    """Linear layer for TIML encoder.

    Args:
        in_features: Number of input features.
        out_shape: Number of putput features.
        sum_from: What number to sum from.
    """

    def __init__(
        self,
        in_features: int,
        out_shape: Tuple[int, int],
        sum_from: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sum_from = sum_from
        self.out_features = out_shape[0] * out_shape[1]
        self.in_features = in_features
        self.out_shape = out_shape

        self.linear = nn.Linear(in_features, self.out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.sum_from is not None:
            x = self.sum_from + torch.sigmoid(x)
        return x.view(x.size(0), *self.out_shape)
