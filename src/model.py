import torch
import torch.nn as nn


class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            # flatten somehow
            nn.Flatten(),
            # Linear: 4096 -> something
            nn.Linear(4096, 512),
            # ReLU
            nn.ReLU(),
            # Linear: something -> 10
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # your forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    model = CIFARClassifier()

    x = torch.randn(64, 3, 32, 32)
    output = model(x)

    print(output.shape)