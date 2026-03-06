import torch
import torch.nn as nn
from torchvision import models

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=5, backbone='densenet121', use_pretrained=True):
        super().__init__()
        weights = "DEFAULT" if use_pretrained else None
        
        if backbone == 'densenet121':
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        return self.backbone(x)  # raw logits, no sigmoid (handled in loss)
