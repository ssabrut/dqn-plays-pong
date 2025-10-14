import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    Implements a standard residual block with two convolutional layers.

    This block applies two 3x3 convolutional layers with batch normalization
    and ReLU activation. A skip connection adds the input of the block to its
    output before the final activation, which helps in training deeper networks.

    Attributes:
        conv1 (nn.Conv2d): The first 2D convolutional layer.
        bn1 (nn.BatchNorm2d): The batch normalization layer following the first convolution.
        conv2 (nn.Conv2d): The second 2D convolutional layer.
        bn2 (nn.BatchNorm2d): The batch normalization layer following the second convolution.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): The number of channels in the input tensor.

        Raises:
            ValueError: If `in_channels` is not a positive integer.
        """
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(
                f"'in_channels' must be a positive integer, but got {in_channels}."
            )

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=in_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the residual block.

        Args:
            x (torch.Tensor): The input tensor, expected to be of shape
                              (N, C, H, W) where C is `in_channels`.

        Returns:
            torch.Tensor: The output tensor after applying the residual block operations.

        Raises:
            TypeError: If the input `x` is not a torch.Tensor.
            ValueError: If the input tensor `x` is not a 4D tensor or if its
                        channel dimension does not match `in_channels`.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(x)}.")
        if x.dim() != 4:
            raise ValueError(
                f"Expected a 4D input tensor (N, C, H, W), but got a tensor with {x.dim()} dimensions."
            )
        if x.size(1) != self.conv1.in_channels:
            raise ValueError(
                f"Input tensor has {x.size(1)} channels, but the block was initialized "
                f"with {self.conv1.in_channels} channels."
            )

        residual: torch.Tensor = x
        out: torch.Tensor = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual

        return F.relu(out)
