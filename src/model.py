import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    """ConvNeXt-Tiny pretrained model for pneumonia classification."""
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    
    # Replace classifier head
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)