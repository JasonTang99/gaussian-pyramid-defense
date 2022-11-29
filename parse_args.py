import argparse
import torch
from torchvision.transforms import InterpolationMode
import os

def parse_args(mode="train"):
    """
    Parse command line arguments. Takes in either "train" or "test" mode.
    """
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')

    # ensemble parameters
    parser.add_argument('--up_samplers', type=int, default=1, help='Number of up-sampling ensemble models.')
    parser.add_argument('--down_samplers', type=int, default=3, help='Number of down-sampling ensemble models.')
    parser.add_argument('--interpolation', type=str, default='bilinear', help='Interpolation method for resizing. \
        One of [bilinear, nearest].')
    parser.add_argument('--scaling_factor', type=float, default=2.0, help='Scaling factor for up/down sampling.')

    if mode == "test":
        parser.add_argument('--voting_method', type=str, default='simple_avg', help='Voting method to use. \
            One of [simple_avg, weighted_avg, majority_vote, weighted_vote].')

    # training related args
    if mode == "train":
        parser.add_argument('--pretrained', action='store_true', help='Whether to start from pretrained ensemble models.')
        parser.add_argument('--archs', type=str, nargs='+', default=['resnet18'], help='List of ensemble model architectures. \
            Must have architectures from [resnet18, resnet34, resnet50]. Must be length 1 or 1 + up_samplers + down_samplers.')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')

    args = parser.parse_args()
    
    # use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check training args
    if mode == "train":
        if any([arch not in ['resnet18', 'resnet34', 'resnet50'] for arch in args.archs]):
            raise ValueError("Model architecture not supported")
        
        if len(args.archs) == 1:
            args.archs = args.archs * (1 + args.up_samplers + args.down_samplers)
        elif len(args.archs) != 1 + args.up_samplers + args.down_samplers:
            raise ValueError("Must specify either one model or one model per ensemble member")
    
    if args.interpolation == 'nearest':
        args.interpolation = InterpolationMode.NEAREST
    elif args.interpolation == 'bilinear':
        args.interpolation = InterpolationMode.BILINEAR
    else:
        raise ValueError("Interpolation method not supported")

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
            'BL' if args.interpolation else 'NN'
        )
        model_paths.append(model_path)
    args.model_paths = model_paths
    
    # dataset specific parameters
    if args.dataset == 'mnist':
        args.input_size = 28
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        args.input_size = 32
        args.num_classes = 10
    else:
        raise ValueError("Dataset not supported")

    # print args
    print("Arguments:")
    print('  ' + '\n  '.join(f'{k:15}= {v}' for k, v in vars(args).items()))

    return args
