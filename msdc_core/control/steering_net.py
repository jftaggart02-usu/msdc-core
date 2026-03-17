"""This module implements the steering control neural network for STIV.
The neural network is based on the DAVE-2 architecture from [1].
The network interprets raw images and generates a steering angle command.

References:
[1] M. Bojarski et al., “End to End Learning for Self-Driving Cars,” 2016. Available: https://arxiv.org/pdf/1604.07316v1
"""

import torch
from torch import nn


class SteeringNet(nn.Module):
    """A neural network based on the DAVE-2 architecture for generating steering angle commands based on raw images."""

    def __init__(self):
        super().__init__()

        self.activation_function = nn.ReLU

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            self.activation_function(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            self.activation_function(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            self.activation_function(),
            nn.Conv2d(48, 64, kernel_size=3),
            self.activation_function(),
            nn.Conv2d(64, 64, kernel_size=3),
            self.activation_function(),
            nn.Flatten(),
        )

        self.fc_layers = nn.Sequential(
            nn.LazyLinear(100),
            self.activation_function(),
            nn.Linear(100, 50),
            self.activation_function(),
            nn.Linear(50, 10),
            self.activation_function(),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the network.

        Args:
            x: Input image tensor
        """

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x
