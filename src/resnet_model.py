import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CIFARResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            10
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = CIFARResNet18()

    x = torch.randn(8, 3, 32, 32)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)