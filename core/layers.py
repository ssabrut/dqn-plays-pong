from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x

        out = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(self.conv2(out))

        out += residual
        return F.relu(out)