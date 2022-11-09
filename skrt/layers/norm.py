import torch
import torch.nn as nn


class VideoTemporalBatchNorm(nn.Module):
    def __init__(self,
                 num_channels: int = 1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_channels)

    def forward(self,
                x: torch.Tensor):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
