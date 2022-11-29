import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import InterpolationMode

import os
from tqdm import tqdm

from utils import create_resnet, calc_resize_shape
from models.gp_ensemble import GPEnsemble 
from parse_args import parse_args
from datasets import load_data

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

def evaluate_baseline(args, model):
    """
    Evaluate model on clean data.
    """
    # load dataset
    test_loader = load_data(args, 0, train=False)

    # run evaluation
    model.eval()
    correct = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    test_acc = correct.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc}')

    return test_acc


def evaluate_FGSM(args, model):
    """
    Evaluate model on FGSM attacked data.
    """
    # load dataset
    test_loader = load_data(args, 0, train=False)

    # run evaluation
    model.eval()
    correct = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        images = fast_gradient_method(
            model_fn=model,
            x=images,
            eps=args.epsilon,
            norm=args.norm,
            clip_min=0.0,
            clip_max=1.0,
        )
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    test_acc = correct.double() / len(test_loader.dataset)
    print(f'Test Accuracy with FGSM Attack: {test_acc}')

    return test_acc

def run_one_attack(args):
    # setup ensemble model
    model = GPEnsemble(args)
    model.eval()

    # run attack
    if args.attack_method == "baseline":
        test_acc = evaluate_baseline(args, model)
    elif args.attack_method == "fgsm":
        test_acc = evaluate_FGSM(args, model)
    

    return test_acc


if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(0)

    # parse args
    args = parse_args(mode="attack")

    # run attack
    run_one_attack(args)
    