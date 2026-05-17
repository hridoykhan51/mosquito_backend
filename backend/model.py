
import torch
import torch.nn as nn
from torchvision import models


class CustomModel(nn.Module):
    def __init__(self, num_classes, hidden_sizes=[256, 128]):
        super(CustomModel, self).__init__()

        self.mobilenet = models.mobilenet_v2(weights=None)

        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Identity()

        layers = []

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(num_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            num_features = hidden_size

        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(num_features, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        features = self.mobilenet(x)
        logits = self.classifier(features)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("weights.pth", map_location=device)

CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE = checkpoint["img_size"]
MEAN = checkpoint["mean"]
STD = checkpoint["std"]
MODEL_NAME = checkpoint.get("model_name", "MobileNetV2 Mosquito Classifier")

model = CustomModel(num_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
