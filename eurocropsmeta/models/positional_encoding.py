import logging
import math
from typing import cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional Encoding for timeseries input vector.

    Args:
        d_hid: Input dimension for Positional Encoding.
        pos_enc_len: Length of positional encoder table.
        t: Period to use for positional encoding
        padding_value_dates: Value used to find padding in dates.

    Raises:
        AssertionError: If dates has more entries than x.
    """

    def __init__(self, d_hid: int, pos_enc_len: int, t: int = 1000, padding_value_dates: int = -1):
        super().__init__()
        timesteps = torch.arange(pos_enc_len).unsqueeze(1)
        # Calculate the positional encoding p
        p = torch.zeros(pos_enc_len, d_hid)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(float(t)) / d_hid))
        p[:, 0::2] = torch.sin(timesteps * div_term)
        p[:, 1::2] = torch.cos(timesteps * div_term)
        self.register_buffer("p", p)
        self.padding_value_dates = padding_value_dates

    def forward(self, x: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        assert dates.size(-1) <= x.size(-2)
        p = cast(torch.Tensor, self.p)
        dates_mask = dates.eq(self.padding_value_dates)

        if dates.size(-1) < x.size(-2):
            adjusted_p = torch.zeros_like(x)
            for batch_idx in range(dates.size(0)):
                # mask out padded indices
                non_padded_mask = ~dates_mask[batch_idx]
                valid_indices = dates[batch_idx][non_padded_mask]
                # Input values from p into the correct positions in adjusted_p
                adjusted_p[batch_idx][valid_indices] = p[valid_indices]
            p = adjusted_p
        else:
            dates[dates_mask] = 0
            p = torch.masked_fill(p[dates], dates_mask.unsqueeze(-1), 0)

        return x + p
