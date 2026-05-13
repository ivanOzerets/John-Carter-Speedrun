import torch
import torch.nn as nn
from torchvision import models

def get_gradual_layers(model_name):
    '''
    Return the layer groups to unfreeze for gradual unfreezing based on the model architecture.
    '''
    layer_map = {
        "resnet18": [["layer4"], ["layer3"], ["layer2"]],
        "resnet50": [["layer4"], ["layer3"], ["layer2"]],
        "resnet101": [["layer4"], ["layer3"], ["layer2"]],
        "mobilenet_v3_small": [["features.12"], ["features.10"], ["features.8"]],
        "mobilenet_v3_large": [["features.16"], ["features.14"], ["features.12"]],
        "efficientnet_b0": [["features.8"], ["features.7"], ["features.6"]],
        "efficientnet_b3": [["features.8"], ["features.7"], ["features.6"]]
    }

    return layer_map[model_name]

class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=3, base_filters=32, num_classes=6, dropout_rate=0.3, in_channels=3):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        out_channels = base_filters

        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            out_channels *= 2

        spatial_size = 224 // (2 ** num_conv_layers)
        final_channels = base_filters * (2 ** (num_conv_layers - 1))
        fc1_input_size = final_channels * spatial_size * spatial_size
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = torch.relu(x)
            x = self.pool(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def get_pretrained_model(model_name, num_classes, dropout_rate=0.5, freeze=True):
    model_registry = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_registry.keys())}")

    constructor, weights = model_registry[model_name]
    model = constructor(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if model_name.startswith("resnet"):
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model

def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True