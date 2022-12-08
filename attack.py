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

from utils import read_results, write_results
from models.gp_ensemble import GPEnsemble 
from parse_args import parse_args, post_process_args, process_args
from datasets import load_data

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

def evaluate_attack(args, linear_model, voting_model=None, denoiser=None):
    """
    Evaluate model on attacked data. Only for baseline, fgsm, pgd attacks.

    Args:
        args: command line arguments
        linear_model: linear model to evaluate, used to generate adversarial examples
        voting_model: voting model to evaluate
        denoiser: denoiser to apply to adversarial examples prior to the ensemble
    """
    # load dataset
    test_loader = load_data(args, 0, train=False)

    # run evaluation
    linear_model.eval()
    if voting_model is not None:
        voting_model.eval()

    if denoiser is not None:
        denoiser.eval()

    linear_correct, voting_correct = 0.0, 0.0
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)

        if args.attack_method == 'baseline':
            pass
        elif args.attack_method == 'fgsm':
            images = fast_gradient_method(
                model_fn=linear_model,
                x=images,
                eps=args.epsilon,
                norm=args.norm,
                clip_min=0.0,
                clip_max=1.0,
            )
        elif args.attack_method == 'pgd':
            images = projected_gradient_descent(
                model_fn=linear_model,
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
        else:
            raise ValueError(f'Invalid attack method: {args.attack_method}')

        with torch.no_grad():
            if denoiser is not None:
                images = denoiser(images)

            linear_out = linear_model(images)
            _, linear_preds = torch.max(linear_out, 1)
            linear_correct += torch.sum(linear_preds == labels).detach().cpu()

            if voting_model is not None:
                voting_out = voting_model(images)
                _, voting_preds = torch.max(voting_out, 1)
                voting_correct += torch.sum(voting_preds == labels).detach().cpu()

    linear_acc = linear_correct / len(test_loader.dataset)
    voting_acc = voting_correct / len(test_loader.dataset)

    return linear_acc, voting_acc

def evaluate_cw_l2(args, linear_model, voting_model=None, denoiser=None, 
        epsilons=[0.5, 1.0, 2.0, 3.5]):
    """
    Evaluate model on attacked data for CW attacks. 
    
    Since CW does not allow the specification of a hard L2 limit in an
    epsilon, we instead run the attack with a small initial constant with
    default parameters and then measure the L2 norm of the perturbations.

    We only count a successful attack if the perturbation is within the 
    epsilon limit and the model prediction changes. We take in a list of
    epsilon limits (epsilons) to evaluate.
    """
    # load dataset
    test_loader = load_data(args, 0, train=False)

    # create buckets to track adversarial examples within each epsilon limit
    linear_correct = [0 for _ in range(len(epsilons))]
    voting_correct = [0 for _ in range(len(epsilons))]
    
    # run evaluation
    linear_model.eval()
    if voting_model is not None:
        voting_model.eval()

    if denoiser is not None:
        denoiser.eval()
    
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)

        adv_images = carlini_wagner_l2(
            model_fn=linear_model,
            x=images,
            n_classes=args.num_classes,
            lr=5e-3,
            binary_search_steps=10,
            max_iterations=150,
            initial_const=args.initial_const,
        )

        # Track the maximum L2 and Linf distances
        l2 = torch.norm((images - adv_images).view(images.shape[0], -1), p=2, dim=1)

        # Check which adversarial examples were successfully found
        with torch.no_grad():
            if denoiser is not None: #add a denoiser
                adv_images = denoiser(adv_images)
                
            linear_out = linear_model(adv_images)
            _, linear_preds = torch.max(linear_out, 1)
            if voting_model is not None:
                voting_out = voting_model(adv_images)
                _, voting_preds = torch.max(voting_out, 1)
            for i, eps in enumerate(epsilons):
                linear_correct[i] += torch.sum((l2 > eps) | (linear_preds == labels)).detach().cpu()
                if voting_model is not None:
                    voting_correct[i] += torch.sum((l2 > eps) | (voting_preds == labels)).detach().cpu()
        print(linear_correct)
        print(voting_correct)
        
    linear_acc = [c / len(test_loader.dataset) for c in linear_correct]
    voting_acc = [c / len(test_loader.dataset) for c in voting_correct]

    return linear_acc, voting_acc


if __name__ == "__main__":
    # parse args
    args = process_args(mode="attack")
    
    args.epsilon = 0.3
    args.up_samplers = 1
    args.down_samplers = 1
    
    args.attack_method = "cw"
    args.batch_size = 32
    
    args = post_process_args(args, mode="attack")

    # setup ensemble model
    model = GPEnsemble(args)

    print(evaluate_cw_l2(args, model, model))