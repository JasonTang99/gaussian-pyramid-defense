import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import InterpolationMode

import numpy as np
import os
from tqdm import tqdm

from utils import create_resnet, calc_resize_shape, read_results, write_results
from models.gp_ensemble import GPEnsemble 
from parse_args import parse_args

def train_one_model(args, model_idx):
    """
    Train one model on resized dataset.

    Arguments:
        args:
            dataset (str): Dataset to train on. One of ['mnist', 'cifar10'].
            input_size (int): Size of input images.
            num_classes (int): Number of classes in dataset.

            model_folder (str): Folder to save model to.
            model_paths (list): List of model paths to save to.

            up_samplers (int): Number of up-sampling ensemble models.
            down_samplers (int): Number of down-sampling ensemble models.
            interpolation (int): Interpolation method to use. 
                One of [InterpolationMode.NEAREST, InterpolationMode.BILINEAR].
            scaling_factor (float): Scaling factor for up/down sampling.

            pretrained (bool): Whether to start from pretrained ensemble models.
            archs (list): List of model architectures to train.
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            lr (float): Learning rate.
            device (str): Device to train on. One of ['cpu', 'cuda:0'].
        
        model_idx (int): Index of model to train.
    """
    # setup transform
    target_size = calc_resize_shape(
        in_size=args.input_size,
        scaling_exp=model_idx - args.down_samplers,
        scaling_factor=args.scaling_factor
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            target_size, 
            interpolation=args.interpolation, 
            antialias=True
        ),
    ])

    # load dataset
    if args.dataset == 'mnist':
        # duplicate to 3 channels
        transform = transforms.Compose([
            *transform.transforms,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        dataset = MNIST(root='data', train=True, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(root='data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported.")
    
    # 80/20 train/validation split
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # setup model
    model = create_resnet(
        arch=args.archs[model_idx],
        num_classes=args.num_classes,
        device=args.device,
        pretrained=args.pretrained
    )
    model.train()
    
    # train model
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    pbar = tqdm(total=len(train_data) * args.epochs // args.batch_size)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            
            loss = loss_fn(output, target)
            loss.backward()
            
            optim.step()
            optim.zero_grad()
            
            pbar.update(1)

        # evaluate model
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_acc = correct / len(val_data)
        print(f"Epoch: {epoch}, Accuracy: {val_acc}")
    pbar.close()

    # save model
    torch.save(model.state_dict(), args.model_paths[model_idx])
    return val_acc


def train_ensemble(args):
    """
    Train ensemble models on a dataset.

    """
    val_accs = {}
    for i, model_path in enumerate(args.model_paths):
        # skip if model already exists
        if os.path.exists(model_path):
            print(f"Model {model_path} already exists. Skipping.")
        else:
            print(f"Training {model_path}.")
            val_acc = train_one_model(args, i)
            val_accs[model_path] = val_acc
    
    # save validation accuracies
    write_results(args.model_folder, val_accs)

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse args
    args = parse_args(mode='train')
    print(args.model_paths)

    # train ensemble
    train_ensemble(args)
    print("Finished training ensemble.")

