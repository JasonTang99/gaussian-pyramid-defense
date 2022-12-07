import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import os
import pickle

# Read file
def read_results(fp):
    if not os.path.exists(fp):
        raise ValueError("Results file not found.")
    
    with open(fp, "rb") as f:
        results = pickle.load(f)
    
    return results

# Write results to the "results" file
# Maps model_path to validation accuracy
def write_results(fp, results, dictionary=True, overwrite=False):
    # Read and update existing results if they exist
    if os.path.exists(fp) and not overwrite:
        past_results = read_results(fp)
        if dictionary:
            past_results.update(results)
        else:
            past_results += results
        results = past_results

    with open(fp, "wb") as f:
        pickle.dump(results, f)

# Calculate target resize shape for a given input size and scaling level
def calc_resize_shape(in_size, scaling_exp, scaling_factor=2):
    return int(in_size * (scaling_factor ** scaling_exp))
