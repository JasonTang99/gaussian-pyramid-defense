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

from utils import create_resnet, calc_resize_shape, read_results, write_results
from models.gp_ensemble import GPEnsemble 
from parse_args import parse_args, post_process_args, process_args
from datasets import load_data

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans_fixed.carlini_wagner_l2 import carlini_wagner_l2
# from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

def evaluate_attack(args, linear_model, voting_model=None, denoiser=None):
    """
    Evaluate model on attacked data. Only for baseline, fgsm, pgd attacks.

    models: [linear_model, voting_model]
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
            if denoiser is not None: #add a denoiser
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

def evaluate_cw_l2(args, linear_model, voting_model=None, denoiser=None, epsilons=[0.5, 1.0, 2.0, 3.5]):
    """
    Evaluate model on attacked data for C&W attacks. 
    
    Since C&W doesn't allow for specified epsilons, we run the attack at a certain setting 
    and report the accuracy at different epsilon limits.
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
            max_iterations=100,
            initial_const=args.initial_const,
        )

        # Track the maximum L2 and Linf distances
        l2 = torch.norm((images - adv_images).view(images.shape[0], -1), p=2, dim=1)

        # Check which adversarial examples were successfully found
        with torch.no_grad():
            if denoiser is not None: #add a denoiser
                images = denoiser(images)
                
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

    # set random seeds
    torch.manual_seed(0)

    # setup ensemble model
    model = GPEnsemble(args)

    print(evaluate_cw_l2(args, model, model))

    # run attack
    # for w in [1e-8]:
    #     for x in [5e-3]:
    #         for y in [10]:
    #             for z in [1000]:
    #                 # if (w, x, y, z) in res:
    #                 #     continue
    #                 print("======================= w", w, "x", x, "y", y, "z", z)
    #                 args.initial_const = w
    #                 correct, l2, linf = evaluate_cw(args, model, x, y, z)
    #                 # res[(w, x, y, z)] = (correct, l2, linf)
    #                 # write_results(fp, res)



# 1e-7 0.5 5 100: 984 0.0655
# 1e-6 0.5 5 100: 587 0.9473
#   0.75 5 70: 357 1.9484
# 1e-5 0.5 5 100: 395 0.9260
#   0.75 5 70: 233 2.2975
# 1e-4 0.5 5 100: 75 tensor(1.4031)
#   0.75 5 70: 131 1.5001

# 1e-6: 191/1024 2.4079
# 1e-5 : 149/1024 2.3400 
#   (with 0.5 lr it gets 0.91 but 325/1024)
# 1e-4 1.0 5 50: 130/1024 1.6033
# 0.001 1.0 5 50: 40/1024 1.9224
# 0.01 1.0 5 50: 5/1024 2.0502 56 seconds
# 0.1 1.0 5 50: 6/1024 2.0107 56 seconds
# 1.0 1.0 5 50: 9/1024 2.7257 56 seconds

# 0.876953125 5e-2 1 600
# 0.849609375 1e-1 1 600
# 1.0 1 50: 0.1474609375 2.4272 (12 seconds)
# 1.0 1 200: 0.1474609375 2.4027 (47 seconds / 512)
# 1.0 1 400: 0.146484375 2.3819 
# 1.0 1 1000: 0.1484375 2.3436

# 1.0 5 25: 9/1024 2.1462 30 seconds
# 1.0 5 50: 5/1024 2.0502 56 seconds
# 1.0 3 50: 130/1024 1.6441 34 seconds
# 1.0 5 200: 5/1024 2.0226 3:39

# 1.0 10 25: 5/1024 2.1640 58 seconds
# 1.0 10 50: 5/1024 2.0833 1:50