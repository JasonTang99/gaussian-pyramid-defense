import argparse
import torch
from torchvision.transforms import InterpolationMode
import os
import numpy as np

def post_process_args(args, mode="train"):
    """
    Post-process and verify arguments.
    """
    # use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check args
    if any([arch not in ['resnet18', 'resnet34', 'resnet50'] for arch in args.archs]):
        raise ValueError("Model architecture not supported")
    
    if len(args.archs) == 1:
        args.archs = args.archs * (1 + args.up_samplers + args.down_samplers)
    elif len(args.archs) != 1 + args.up_samplers + args.down_samplers:
        raise ValueError("Must specify either one model or one model per ensemble member")
    
    if mode == "attack":
        if args.norm == 'inf':
            args.norm = np.inf
        elif args.norm == '1' or args.norm == '2':
            args.norm = int(args.norm)
        else:
            raise ValueError("Norm not supported")
    


    # generate model folder
    args.model_folder = os.path.join('trained_models', args.dataset, "")
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    
    # generate model paths
    model_paths = []
    for i, arch in enumerate(args.archs):
        scaling_exp = i - args.down_samplers
        model_path = args.model_folder + "{}_{}{}{}_{}.pth".format(
            arch,
            args.scaling_factor,
            '+' if scaling_exp >= 0 else '',
            scaling_exp,
            'BL' if args.interpolation == 'bilinear' else 'NN'
        )
        model_paths.append(model_path)
    args.model_paths = model_paths
    
    if args.interpolation == 'nearest':
        args.interpolation = InterpolationMode.NEAREST
        args.antialias = False
    elif args.interpolation == 'bilinear':
        args.interpolation = InterpolationMode.BILINEAR
        args.antialias = True
    else:
        raise ValueError(f"Interpolation method {args.interpolation} not supported")

    # dataset specific parameters
    if args.dataset == 'mnist':
        args.input_size = 28
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        args.input_size = 32
        args.num_classes = 10
    else:
        raise ValueError("Dataset not supported")

    return args


def process_args(mode="train"):
    """
    Parse command line arguments. Takes in either "train" or "attack" mode.
    """
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')

    # ensemble parameters
    parser.add_argument('--up_samplers', type=int, default=0, help='Number of up-sampling ensemble models.')
    parser.add_argument('--down_samplers', type=int, default=0, help='Number of down-sampling ensemble models.')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='Interpolation method for resizing. \
        One of [bilinear, nearest].')
    parser.add_argument('--scaling_factor', type=float, default=2.0, help='Scaling factor for up/down sampling.')
    parser.add_argument('--archs', type=str, nargs='+', default=['resnet18'], help='List of ensemble model architectures. \
        Must have architectures from [resnet18, resnet34, resnet50]. Must be length 1 or 1 + up_samplers + down_samplers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
        
    if mode == "attack":
        parser.add_argument('--voting_method', type=str, default='simple_avg', help='Voting method to use. \
            One of [simple_avg, weighted_avg, majority_vote, weighted_vote].')
        parser.add_argument('--attack_method', type=str, default='fgsm', help='Attack method to use. \
            One of [baseline, fgsm, pgd, cw].')
        # general attack parameters
        parser.add_argument('--epsilon', type=float, default=0.3, help='Epsilon value for attack.')
        parser.add_argument('--norm', type=str, default='2', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')

        # pgd attack parameters
        parser.add_argument('--nb_iter', type=int, default=40, help='Number of steps for PGD attack. Usually 40 or 100.')
        parser.add_argument('--eps_iter', type=float, default=0.01, help='Step size for PGD attack.')
        parser.add_argument('--rand_init', type=bool, default=False, help='Whether to use random initialization for PGD attack.')
        
        # cw attack parameters
        parser.add_argument('--initial_const', type=float, default=0.01, help='Initial constant for CW attack. Lower values generally result in smaller L2 norms.')

    # training related args
    if mode == "train":
        parser.add_argument('--pretrained', action='store_true', help='Whether to start from pretrained ensemble models.')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')

    args = parser.parse_args()

    return args


def parse_args(mode="train", verbose=True):
    """
    Combines both above functions to collect and parse args.
    """

    args = process_args(mode)
    args = post_process_args(args, mode)

    # print args
    if verbose:
        print("Arguments:")
        print('  ' + '\n  '.join(f'{k:15}= {v}' for k, v in vars(args).items()))
    
    return args
