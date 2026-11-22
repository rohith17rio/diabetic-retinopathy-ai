
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RetinalResNet18(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.network = resnet18(weights=weights)

        # Freeze the pretrained layers initially.
        for param in self.network.parameters():
            param.requires_grad = False

        # Replace the original 1000-class output with 5 retinal grades.
        input_features = self.network.fc.in_features
        self.network.fc = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.network(x)
