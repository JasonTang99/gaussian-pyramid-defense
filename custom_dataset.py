import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
# from torchvision.models import resnet18
from utils import create_resnet

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdversarialDataset(Dataset):
    def __init__(self, args, root='./custom_data', train=True):
        if train:
            gt = f"{args.dataset}_train.pt"
            adv = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_train.pt"
        else:
            gt = f"{args.dataset}_test.pt"
            adv = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_test.pt"

        print(adv)
        self.clean = torch.load(os.path.join(root, gt))
        self.noisy = torch.load(os.path.join(root, adv))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):        
        return self.clean[idx], self.noisy[idx]


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0., 1.)

def get_dataset(args):
    if args.dataset == 'mnist':
        # duplicate to 3 channels
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_data = MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)     
    else:
        raise ValueError("Dataset not supported.")
    
    return train_data, test_data

def get_dataloader(args, val=False):
    train_data, test_data = get_dataset(args)

    if val:
        # 80/20 train/validation split
        train_data, val_data = torch.utils.data.random_split(train_data, [0.8, 0.2])

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        return train_loader, test_loader

def generate_attack(args, model, images):
    # generate random noise level within given range
    eps_random = np.random.uniform(args.eps_range[0], args.eps_range[-1])
    if args.adv_mode == 'fgsm':
        images = fast_gradient_method(
            model_fn=model,
            x=images,
            eps=eps_random,
            norm=args.norm,
            clip_min=0.0,
            clip_max=1.0,
        )
    elif args.adv_mode == 'pgd':
        images = projected_gradient_descent(
            model_fn=model,
            x=images,
            eps=eps_random,
            eps_iter=0.01,
            nb_iter=40,
            norm=args.norm,
            clip_min=0.0,
            clip_max=1.0,
            sanity_checks=False
        )
    elif args.adv_mode == 'cw':
        images = carlini_wagner_l2(
            model_fn=model,
            x=images,
            n_classes=10,
            max_iterations=10
        )

    return images


def generate_adv_examples(args, model):

    train_data, test_data = get_dataset(args)

    # will have one single batch
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_list = []
    test_list = []

    if args.adv_mode == 'none':
        train_file_name = f"{args.dataset}_train.pt"
        test_file_name = f"{args.dataset}_test.pt"

        for (images, _) in train_loader: train_list.append(images)
        for (images, _) in test_loader: test_list.append(images)
    else:
        train_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_train.pt"
        test_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_test.pt"
        # convert to int
        if args.norm == 'inf':
            args.norm = np.inf
        elif args.norm == '1' or args.norm == '2':
            args.norm = int(args.norm)
        else:
            raise ValueError("Norm not supported")
        # generate attack examples    
        for (images, _) in tqdm(train_loader):
            images = images.to(device)
            images = generate_attack(args, model, images)
            train_list.append(images.detach().cpu())
        
        for (images, _) in tqdm(test_loader):
            images = images.to(device)
            images = generate_attack(args, model, images)
            test_list.append(images.detach().cpu())
        
    train_tensor = torch.cat(train_list, dim=0)
    test_tensor = torch.cat(test_list, dim=0)

    if not os.path.isdir('custom_data'):
        os.mkdir('custom_data')

    torch.save(train_tensor, os.path.join("custom_data", train_file_name))
    torch.save(test_tensor, os.path.join("custom_data", test_file_name))

    return

def peak_dataset(gt, noisy, idx):
    tensor1 = torch.load(gt)
    tensor2 = torch.load(noisy)
    img1 = img_to_numpy(tensor1[idx])
    img2 = img_to_numpy(tensor2[idx])
    plt.subplot(211)
    plt.imshow(img1)
    plt.subplot(212)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--adv_mode', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')

    parser.add_argument('--eps_range', type=float, nargs='+', default=[0.25, 3], help='noise level range, sigma for gaussian, eps for adversarial')
    parser.add_argument('--norm', type=str, default='2', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')

    # # pgd attack parameters
    # parser.add_argument('--nb_iter', type=int, default=40, help='Number of steps for PGD attack. Usually 40 or 100.')
    # parser.add_argument('--eps_iter', type=float, default=0.01, help='Step size for PGD attack.')
    # parser.add_argument('--rand_init', type=bool, default=False, help='Whether to use random initialization for PGD attack.')
    # # cw attack parameters
    # parser.add_argument('--confidence', type=float, default=0, help='Confidence for CW attack.')

    args = parser.parse_args()

    # classification model
    net = create_resnet(device=device)
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_2.0+0_BL.pth'), map_location=device))

    print("Generating Adversarial Examples:")
    print(f"dataset: {args.dataset}")
    print(f"attack: {args.adv_mode}")
    print(f"norm: {args.norm}")
    print(f"eps range: {args.eps_range}")
    print("=======================================================")
    generate_adv_examples(args, net)

    test = AdversarialDataset(args, train=False)
    print(test[0])
    plt.imshow(img_to_numpy(test[0][1]))
    plt.show()
    #peak_dataset(os.path.join("custom_data", "mnist_train.pt"), os.path.join("custom_data", "mnist_cw_norm2_train.pt"), 100)
