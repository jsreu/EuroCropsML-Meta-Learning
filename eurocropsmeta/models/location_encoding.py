import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LocationEncodingTIML(nn.Module):
    """Location Encoding for time series input vector for TIML algorithm."""

    @staticmethod
    def three_dimensional_points(latlons: torch.Tensor) -> torch.Tensor:
        # copied from Presto
        with torch.no_grad():
            lats, lons = latlons[:, 0], latlons[:, 1]

            lat_radian = lats * math.pi / 180
            lon_radian = lons * math.pi / 180
            x = torch.cos(lat_radian) * torch.cos(lon_radian)
            y = torch.cos(lat_radian) * torch.sin(lon_radian)
            z = torch.sin(lat_radian)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        task_info = self.three_dimensional_points(centers).to(torch.float)
        return task_info
