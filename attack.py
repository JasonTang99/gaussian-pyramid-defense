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
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2


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


def evaluate_attack(args, model):
    """
    Evaluate model on attacked data.
    """
    # load dataset
    test_loader = load_data(args, 0, train=False)

    # run evaluation
    model.eval()
    correct = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(args.device), labels.to(args.device)

        if args.attack_method == 'baseline':
            pass
        elif args.attack_method == 'fgsm':
            images = fast_gradient_method(
                model_fn=model,
                x=images,
                eps=args.epsilon,
                norm=args.norm,
                clip_min=0.0,
                clip_max=1.0,
            )
        elif args.attack_method == 'pgd':
            images = projected_gradient_descent(
                model_fn=model,
                x=images,
                eps=args.epsilon,
                eps_iter=args.eps_iter,
                nb_iter=args.nb_iter,
                norm=args.norm,
                clip_min=0.0,
                clip_max=1.0,
                rand_init=args.rand_init,
                sanity_checks=False
            )
        elif args.attack_method == 'cw':
            images = carlini_wagner_l2(
                model_fn=model,
                x=images,
                n_classes=args.num_classes,
            )

        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).detach().cpu()

    test_acc = correct.double() / len(test_loader.dataset)
    print(f'Test Accuracy on {args.attack_method}: {test_acc}')

    return test_acc

def run_one_attack(args):
    # set random seeds
    torch.manual_seed(0)

    # setup ensemble model
    model = GPEnsemble(args)

    # run attack
    test_acc = evaluate_attack(args, model)

    return test_acc


if __name__ == "__main__":
    # parse args
    args = parse_args(mode="attack")
    args.attack_method = "cw"

    # run attack
    run_one_attack(args)
    