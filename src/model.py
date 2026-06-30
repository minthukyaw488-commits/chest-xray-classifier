"""
Model architecture definitions for chest X-ray pneumonia classification.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """
    Create a ConvNeXt-Tiny model with custom classifier head.
    
    Args:
        num_classes (int): Number of output classes. Default is 2 
                          (NORMAL, PNEUMONIA).
    
    Returns:
        torch.nn.Module: ConvNeXt-Tiny model with custom classifier.
    """
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    
    # Replace the classifier head for our task
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model


if __name__ == "__main__":
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")