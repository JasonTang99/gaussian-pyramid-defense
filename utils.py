import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import os
import pickle

# Read past results from the "results" file
def read_results(folder_path):
    if not os.path.exists(os.path.join(folder_path, "results")):
        raise ValueError("Results file not found.")
    
    with open(os.path.join(folder_path, "results"), "rb") as f:
        results = pickle.load(f)
    
    return results

# Write results to the "results" file
# Maps model_path to validation accuracy
def write_results(folder_path, results):
    # Read and update existing results if they exist
    if os.path.exists(os.path.join(folder_path, "results")):
        past_results = read_results(folder_path)
        past_results.update(results)
        results = past_results

    with open(os.path.join(folder_path, "results"), "wb") as f:
        pickle.dump(results, f)


# Calculate target resize shape for a given input size and scaling level
def calc_resize_shape(in_size, scaling_exp, scaling_factor=2):
    return int(in_size * (scaling_factor ** scaling_exp))

# Generate resnet model
def create_resnet(arch="resnet18", num_classes=10, device="cpu", pretrained=True):
    """
    Create resnet model.
    """
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 
                         if pretrained else None)
    elif arch == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1
                         if pretrained else None)
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1
                         if pretrained else None)
    else:
        raise ValueError("Model not supported.")
    
    # change output size
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # freeze all non-fc layers
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    
    # move model to device
    model = model.to(device)
    
    return model