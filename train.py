import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import skimage.io
import argparse
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args):
    """
    Train ensemble models on a dataset.

    args:
        dataset (str): Dataset to train on. One of ['mnist', 'cifar10'].
        up_samplers (int): Number of up-sampling ensemble models.
        down_samplers (int): Number of down-sampling ensemble models.
        models (list): List of ensemble models. Must consist of models from [resnet18, resnet34, resnet50]. 
            Must be of length 1 + up_samplers + down_samplers.
        pretrained (bool): Whether to start from pretrained ensemble models.
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    """

    # load data and split into train and validation sets
    if args.dataset == 'mnist':
        train_data = MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])
    elif args.dataset == 'cifar10':
        train_data = CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())
        train_data, val_data = torch.utils.data.random_split(train_data, [40000, 10000])
    else:
        raise ValueError("Dataset not supported")
    
    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    

    # generate model names
    model_names = args.models.copy()
    idx = 0
    for i in range(args.up_samplers):
        model_names[idx] = f"up_{i}_{model_names[idx]}"
        idx += 1
    model_names[idx] = f"base_{model_names[idx]}"
    idx += 1
    for i in range(args.down_samplers):
        model_names[idx] = f"down_{i}_{model_names[idx]}"
        idx += 1
    
    print(model_names)
    print(args.models)
    exit(0)

    # check if models exist
    trained_model_path = os.path.join('trained_models', dataset)
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    


    # create model

    model = DnCNN(use_bias=use_bias, hidden_channels=hidden_channels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)


    losses = []
    psnrs = []
    baseline_psnrs = []
    val_losses = []
    val_psnrs = []
    val_iters = []
    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample in train_dataloader:

            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            loss = torch.mean((denoised_sample - sample)**2)
            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # plot results
            if not idx % plot_every:
                plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                             val_losses, val_psnrs, val_iters, train_dataset,
                             val_dataset, val_dataloader)

            idx += 1
            pbar.update(1)

    pbar.close()
    return model

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--up_samplers', type=int, default=0, help='Number of up-sampling ensemble models.')
    parser.add_argument('--down_samplers', type=int, default=1, help='Number of down-sampling ensemble models.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to start from pretrained ensemble models.')

    # either specify individual models or the same model for all ensemble members
    parser.add_argument('--models', type=str, nargs='+', default=['resnet18'], help='List of ensemble models. Must consist of models from [resnet18, resnet34, resnet50]. Must be of length 1 or 1 + up_samplers + down_samplers.')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    args = parser.parse_args()

    # check args
    if (len(args.models) != 1 + args.up_samplers + args.down_samplers) and (len(args.models) != 1):
        raise ValueError("Must specify either one model or one model per ensemble member")
    if any([model not in ['resnet18', 'resnet34', 'resnet50'] for model in args.models]):
        raise ValueError("Model not supported")
    
    if len(args.models) == 1:
        args.models = args.models * (1 + args.up_samplers + args.down_samplers)

    # train ensemble
    train(args)




