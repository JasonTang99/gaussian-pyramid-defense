import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import MNIST, CIFAR10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import skimage.io
import argparse
from tqdm import tqdm

from models.utils import create_resnet

def calc_target_size(in_size, direction, num, factor=2):
    return in_size * factor**num if direction == "up" else in_size // factor**num

def train_one_model(args, model_name, model, direction, num):
    """
    Train one model on resized dataset.
    """
    
    # load data and split into train and validation sets
    if args.dataset == 'mnist':
        target_size = calc_target_size(28, direction, num)
        # duplicate to 3 channels
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_data = MNIST(root='data', train=True, download=True, transform=transform)
        train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])

    elif args.dataset == 'cifar10':
        target_size = calc_target_size(32, direction, num)
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        train_data = CIFAR10(root='data', train=True, download=True, transform=transform)
        train_data, val_data = torch.utils.data.random_split(train_data, [40000, 10000])
    else:
        raise ValueError("Dataset not supported.")
    
    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # load model
    model = create_resnet(device=args.device, output_size=10, model=model)
    print(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # train model
    pbar = tqdm(total=len(train_data) * args.epochs // args.batch_size)
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            pbar.update(1)

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print("Epoch: {}, Accuracy: {}".format(epoch, correct / len(val_data)))
    pbar.close()

    # save model
    torch.save(model.state_dict(), os.path.join('trained_models', args.dataset, model_name + ".pth"))


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
    # get/setup model path
    trained_model_path = os.path.join('trained_models', args.dataset)
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)

    # generate model names and train untrained models
    model_names = []
    for i, model in enumerate(args.models):
        if i < args.up_samplers:
            model_names.append(f"up_{i+1}_{model}")
            if not os.path.exists(os.path.join(trained_model_path, model_names[-1])):
                train_one_model(args, model_name=model_names[-1], model=model, direction="up", num=i+1)
        elif i == args.up_samplers:
            model_names.append(f"base_{model}")
            if not os.path.exists(os.path.join(trained_model_path, model_names[-1])):
                train_one_model(args, model_name=model_names[-1], model=model, direction="up", num=0)
        else:
            model_names.append(f"down_{i-args.up_samplers}_{model}")
            if not os.path.exists(os.path.join(trained_model_path, model_names[-1])):
                train_one_model(args, model_name=model_names[-1], model=model, direction="down", num=i-args.up_samplers)
    
    print(model_names)
    exit(0)
    


    # create model

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
    parser.add_argument('--up_samplers', type=int, default=2, help='Number of up-sampling ensemble models.')
    parser.add_argument('--down_samplers', type=int, default=3, help='Number of down-sampling ensemble models.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to start from pretrained ensemble models.')

    # either specify individual models or the same model for all ensemble members
    parser.add_argument('--models', type=str, nargs='+', default=['resnet18'], help='List of ensemble models. \
        Must consist of models from [resnet18, resnet34, resnet50]. Must be of length 1 or 1 + up_samplers + down_samplers.')

    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    args = parser.parse_args()
    
    # use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check args
    if (len(args.models) != 1 + args.up_samplers + args.down_samplers) and (len(args.models) != 1):
        raise ValueError("Must specify either one model or one model per ensemble member")
    if any([model not in ['resnet18', 'resnet34', 'resnet50'] for model in args.models]):
        raise ValueError("Model not supported")
    
    if len(args.models) == 1:
        args.models = args.models * (1 + args.up_samplers + args.down_samplers)

    # print args
    print("Arguments:")
    print('  ' + '\n  '.join(f'{k:15}= {v}' for k, v in vars(args).items()))

    # train ensemble
    train(args)




