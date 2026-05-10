import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))
        x = self.conv3(x)
        x = self.pool(torch.relu(x))
        x = self.fc1(torch.flatten(x, 1))
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)
        return x

def get_resnet50(num_classes, freeze=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True