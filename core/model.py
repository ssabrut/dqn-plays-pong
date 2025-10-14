from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from core.layers import ResidualBlock


class DQN(nn.Module):
    """
    Implements a Dueling Deep Q-Network with Residual Blocks.

    This network architecture processes image-based observations through a series of
    convolutional layers and residual blocks. It then uses a dueling structure,
    splitting the output into a value stream and an advantage stream, to produce
    the final Q-value estimates for each action.

    Attributes:
        input_shape (Tuple[int, int, int]): The shape of the input observations (C, H, W).
        num_actions (int): The number of possible actions in the action space.
        conv (nn.Sequential): The initial convolutional layers for feature extraction.
        resnet_blocks (nn.Sequential): A sequence of residual blocks for deeper feature processing.
        advantage_stream (nn.Sequential): The network stream for calculating action advantages.
        value_stream (nn.Sequential): The network stream for calculating the state value.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int) -> None:
        """
        Initializes the DQN model.

        Args:
            input_shape (Tuple[int, int, int]): The shape of the input state, formatted as (channels, height, width).
            num_actions (int): The number of possible discrete actions.

        Raises:
            TypeError: If `input_shape` is not a tuple or `num_actions` is not an integer.
            ValueError: If `input_shape` does not contain three positive integers, or if `num_actions` is not a positive integer.
        """
        super().__init__()
        # --- Validation ---
        if not isinstance(input_shape, tuple):
            raise TypeError(f"input_shape must be a tuple, but got {type(input_shape)}.")
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must have 3 dimensions (C, H, W), but got {len(input_shape)}.")
        if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise ValueError(f"All dimensions in input_shape must be positive integers, but got {input_shape}.")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be a positive integer, but got {num_actions}.")
        # --- End Validation ---

        self.input_shape: Tuple[int, int, int] = input_shape
        self.num_actions: int = num_actions

        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.resnet_blocks: nn.Sequential = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        conv_out_size: int = self._get_conv_out(self.input_shape)
        self.advantage_stream: nn.Sequential = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, self.num_actions)
        )

        self.value_stream: nn.Sequential = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape: Tuple[int, int, int]) -> int:
        """
        Calculates the output size of the convolutional and residual layers.

        This helper function determines the input size for the fully connected layers
        by performing a forward pass with a dummy tensor of the specified shape.

        Args:
            shape (Tuple[int, int, int]): The shape of the input tensor (C, H, W).

        Returns:
            int: The total number of features after the convolutional and residual layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            o: torch.Tensor = self.conv(dummy_input)
            o = self.resnet_blocks(o)
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        The input tensor is first normalized from [0, 255] to [0.0, 1.0]. It is then
        passed through the convolutional and residual layers, and finally through the
        dueling streams to compute Q-values.

        Args:
            x (torch.Tensor): The input batch of states, expected to be of shape
                              (N, C, H, W) matching the `input_shape`.

        Returns:
            torch.Tensor: The calculated Q-values for each action, with shape (N, num_actions).

        Raises:
            TypeError: If the input `x` is not a torch.Tensor.
            ValueError: If `x` is not a 4D tensor or if its shape (excluding the batch
                        dimension) does not match the model's `input_shape`.
        """
        # --- Validation ---
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(x)}.")
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D input tensor (N, C, H, W), but got a tensor with {x.dim()} dimensions.")
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input tensor shape {x.shape[1:]} does not match the expected "
                f"model input shape {self.input_shape}."
            )
        # --- End Validation ---

        x = x.float() / 255.0
        conv_out: torch.Tensor = self.conv(x)
        resnet_out: torch.Tensor = self.resnet_blocks(conv_out)

        flattened_out: torch.Tensor = resnet_out.view(x.size(0), -1)

        value: torch.Tensor = self.value_stream(flattened_out)
        advantage: torch.Tensor = self.advantage_stream(flattened_out)

        q_values: torch.Tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values