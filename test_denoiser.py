import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from models.denoisers import DnCNN, REDNet20
from custom_dataset import AdversarialDataset, get_dataloader
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from train_denoiser import img_to_numpy
from utils import create_resnet

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_batch(org, adv, denoised, n=6):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(img_to_numpy(org[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display noisy
        ax = plt.subplot(3, n, i+1 + n)
        img = img_to_numpy(adv[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display denoised image
        ax = plt.subplot(3, n, i+1 + n + n)
        img = img_to_numpy(denoised[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5,0.65, "ADV IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5,0.35, "DENOISED IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace = 0.5)    
    plt.show()

def test_acc(model, denoiser, test_loader, eps, norm, attack):
    model.eval()
    accuracy1 = 0.0
    accuracy2 = 0.0
    accuracy3 = 0.0
    total = 0.0
    
    # with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        if attack == 'fgsm':
            x_adv = fast_gradient_method(
                model_fn=model,
                x=images,
                eps=eps,
                norm=norm,
                clip_min=0.0,
                clip_max=1.0,
            )
        elif attack == 'pgd':
            x_adv = projected_gradient_descent(
                model_fn=model,
                x=images,
                eps=eps,
                eps_iter=0.01,
                nb_iter=20,
                norm=norm,
                clip_min=0.0,
                clip_max=1.0,
                sanity_checks=False
            )
        elif attack == 'cw':
            x_adv = carlini_wagner_l2(
                model_fn=model,
                x=images,
                n_classes=10,
                max_iterations=10
            )
        else: pass

        # baseline
        outputs = model(images)
        # attack
        adv_outputs = model(x_adv)
        # attack denoised
        denoised = denoiser(x_adv)
        denoised_outputs = model(denoised)

        _, pred = torch.max(outputs, 1)
        _, pred_adv = torch.max(adv_outputs, 1)
        _, pred_dn = torch.max(denoised_outputs, 1)

        total += labels.size(0)
        accuracy1 += (pred == labels).sum().item()
        accuracy2 += (pred_adv == labels).sum().item()
        accuracy3 += (pred_dn == labels).sum().item()
    
    show_batch(images, x_adv, denoised, n=10)

    # compute the accuracy over all test images
    accuracy1 = (accuracy1 / total)
    accuracy2 = (accuracy2 / total)
    accuracy3 = (accuracy3 / total)
    print("Test Accuracy no attack: {}".format(accuracy1))
    print("Test Accuracy with {} attack: {}".format(attack, accuracy2))
    print("Test Accuracy with {} attack + denoiser: {}".format(attack, accuracy3))

    return accuracy1, accuracy2, accuracy3


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--arch', type=str, default='dncnn', help='Dataset to train on. One of [dncnn, dae1, dae2].')
    parser.add_argument('--denoiser', type=str, default='gaussian', help='')
    parser.add_argument('--adv_mode', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--eps', type=float, default=3, help='perturbation level')
    parser.add_argument('--norm', type=str, default='2', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    args = parser.parse_args()

    # load data
    #if adv_mode = "cw"
    _, test_loader = get_dataloader(args.dataset, args.batch_size, val=False)

    # denoiser model
    if args.arch == 'dncnn':
        denoiser = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)
    elif args.arch == 'dae':
        denoiser = REDNet20(in_channels=3, out_channels=3, use_bias=False).to(device)

    # load pretrained denoiser
    denoiser_name = f"{args.arch}_{args.dataset}_{args.denoiser}.pth"
    denoiser_path = './trained_denoisers/' + denoiser_name
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))

    #load classification model
    net = create_resnet(device=device)
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_2.0+0_BL.pth'), map_location=device))


    if args.norm == 'inf':
        args.norm = np.inf
    elif args.norm == '1' or args.norm == '2':
        args.norm = int(args.norm)
    else:
        raise ValueError("Norm not supported")
    
    print("=================== Testing ====================")
    print(f"denoiser: {denoiser_name}")
    print(f"dataset: {args.dataset}")
    print(f"noise type: {args.adv_mode}")
    print(f"eps: {args.eps}")
    print(f"norm: {args.norm}")
    print("=======================================================")
    test_acc(net, denoiser, test_loader, eps=args.eps, norm=args.norm, attack=args.adv_mode)
