import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

def create_resnet(device="cpu", output_size=10, model="resnet18"):
    """
    Create resnet model.
    """
    if model == "resnet18":
        model = resnet18(weights=ResNet18_Weights)
    elif model == "resnet34":
        model = resnet34(weights=ResNet34_Weights)
    elif model == "resnet50":
        model = resnet50(weights=ResNet50_Weights)
    else:
        raise ValueError("Model not supported.")
    
    # change output size
    model.fc = nn.Linear(model.fc.in_features, output_size)

    # freeze all non-fc layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model