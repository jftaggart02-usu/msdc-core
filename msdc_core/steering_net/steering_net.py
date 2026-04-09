"""This module implements the SteeringNet class, which is a neural network architecture designed for steering angle prediction in autonomous driving applications.
The network takes in images from the vehicle's camera and outputs a predicted steering angle.
The network is based on NVIDIA's DAVE-2 architecture.
"""

import torch


class SteeringNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input image size (3, 240, 320) - RGB image with height 240 and width 320

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = torch.nn.LazyLinear(out_features=100)
        self.fc2 = torch.nn.Linear(in_features=100, out_features=50)
        self.fc3 = torch.nn.Linear(in_features=50, out_features=10)
        self.fc4 = torch.nn.Linear(in_features=10, out_features=1)

        self.layers = torch.nn.Sequential(
            self.conv1,
            torch.nn.ReLU(),
            self.conv2,
            torch.nn.ReLU(),
            self.conv3,
            torch.nn.ReLU(),
            self.conv4,
            torch.nn.ReLU(),
            self.conv5,
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.ReLU(),
            self.fc3,
            torch.nn.ReLU(),
            self.fc4,
        )

    def forward(self, x):
        return self.layers(x)
