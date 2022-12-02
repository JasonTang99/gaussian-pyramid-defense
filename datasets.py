import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import InterpolationMode

from utils import *

def load_data(args, scaling_exp=0, train=True):
    # setup transform
    if train:
        target_size = calc_resize_shape(
            in_size=args.input_size,
            scaling_exp=scaling_exp,
            scaling_factor=args.scaling_factor
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                target_size, 
                interpolation=args.interpolation, 
                antialias=args.antialias
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # load dataset
    if args.dataset == 'mnist':
        # duplicate to 3 channels
        transform = transforms.Compose([
            *transform.transforms,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        dataset = MNIST(root='data', train=train, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(root='data', train=train, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported.")
    
    if train:
        # 80/20 train/validation split
        train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
        
        # create dataloaders
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader
    else:
        # create dataloader
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        return test_loader