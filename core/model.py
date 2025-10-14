import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, Any

from core.layers import ResidualBlock
class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.resnet_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape: Any) -> int:
        o = self.conv(torch.zeros(1, *shape))
        o = self.resnet_blocks(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.
        conv_out = self.conv(x)
        resnet_out = self.resnet_blocks(conv_out)

        flattened_out = resnet_out.view(x.size()[0], -1)

        value = self.value_stream(flattened_out)
        advantage = self.advantage_stream(flattened_out)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values