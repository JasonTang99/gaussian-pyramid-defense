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
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from train_denoiser import img_to_numpy
from utils import create_resnet

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_attack(images, model, attack, norm, eps):
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
            eps_iter=5e-4,
            nb_iter=40,
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
            lr=5e-3,
            binary_search_steps=5,
            max_iterations=100,
            initial_const=1e-3
        )
        l2_norm = torch.norm((images - x_adv).view(images.shape[0], -1), p=2, dim=1)
        indices = (l2_norm > eps)
        # print(x_adv[indices].size())
        # ignore images with l2 norm larger than eps
        x_adv[indices] = images[indices]

    else: x_adv = images

    return x_adv

def show_random_batch(model, denoiser, test_loader, attack, norm, eps, n=5, eval=False):
    denoiser.eval()
    images, _ = next(iter(test_loader))
    # original
    images = images.to(device)
    # adversarial
    x_adv = generate_attack(images, model, attack, norm, eps)
    # denoised
    denoised = denoiser(x_adv)

    plt.figure(figsize=(10, 9), dpi=500)
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(img_to_numpy(images[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display noisy
        ax = plt.subplot(3, n, i+1 + n)
        img = img_to_numpy(x_adv[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display denoised image
        ax = plt.subplot(3, n, i+1 + n + n)
        img = img_to_numpy(denoised[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5,0.95, "Original Images", ha="center", va="top", fontsize=20, color="b")
    plt.figtext(0.5,0.65, f"Adversarial Images ({attack})", ha="center", va="top", fontsize=20, color="r")
    plt.figtext(0.5,0.33, "Denoised Images", ha="center", va="top", fontsize=20, color="g")
    if eval:
        avg_psnr_bl, avg_ssim_bl, avg_psnr, avg_ssim, total = evaluate_metrics(model, denoiser, test_loader, attack, norm, eps)
        plt.figtext(0.5,0.38, "PSNR: {:.3f}dB, SSIM: {:.3f} (averaged over {} images)".format(avg_psnr_bl, avg_ssim_bl, total), ha="center", va="top", fontsize=20)
        plt.figtext(0.5,0.08, "PSNR: {:.3f}dB, SSIM: {:.3f} (averaged over {} images)".format(avg_psnr, avg_ssim, total), ha="center", va="top", fontsize=20)
    plt.subplots_adjust(hspace = 0.1)    
    plt.tight_layout()
    plt.show()

def evaluate_metrics(model, denoiser, test_loader, attack, norm, eps):
    denoiser.eval()
    avg_psnr_bl=0
    avg_ssim_bl=0
    avg_psnr=0
    avg_ssim=0
    total=0
    
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        x_adv = generate_attack(images, model, attack, norm, eps)

        with torch.no_grad():
            output = denoiser(x_adv)

            for i in range(len(images)):
                original = img_to_numpy(images[i])
                adv = img_to_numpy(x_adv[i])
                denoised = img_to_numpy(output[i])
                avg_psnr_bl += PSNR(original, adv)
                avg_ssim_bl += SSIM(original, adv, multichannel=True)
                avg_psnr += PSNR(original, denoised)
                avg_ssim += SSIM(original, denoised, multichannel=True)

            total += len(images)
    avg_psnr_bl /= total      
    avg_ssim_bl /= total  
    avg_psnr /= total
    avg_ssim /= total
    print("\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr, avg_ssim))

    return avg_psnr_bl, avg_ssim_bl, avg_psnr, avg_ssim, total

def test_acc(model, denoisers, test_loader, attack, norm, eps):
    #resnet model
    model.eval()
    #denoisers to test
    for denoiser in denoisers: denoiser.eval()

    n = 2 + len(denoisers) 
    acc_list = np.zeros((n))
    total = 0.0
    
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        # generate attack
        x_adv = generate_attack(images, model, attack, norm, eps)
        
        total += labels.size(0)

        with torch.no_grad():
            # baseline
            outputs = model(images)
            # attack
            adv_outputs = model(x_adv)

            _, pred = torch.max(outputs, 1)
            _, pred_adv = torch.max(adv_outputs, 1)

            acc_list[0] += (pred == labels).sum().item()
            acc_list[1] += (pred_adv == labels).sum().item()

            #denoisers
            for i, denoiser in enumerate(denoisers):
                denoised = denoiser(x_adv)
                denoised_outputs = model(denoised)
                _, pred_dn = torch.max(denoised_outputs, 1)
                acc_list[i+2] += (pred_dn == labels).sum().item()

    # if attack == 'cw': show_batch(images, x_adv, denoised, n=10)

    # compute the accuracy over all test images
    acc_list = (acc_list / total)
    print("Test Accuracy no attack: {}".format(acc_list[0]))
    print("Test Accuracy with {} attack: {}".format(attack, acc_list[1]))
    for i in range(len(denoisers)):
        print("Test Accuracy with {} attack + denoiser{}: {}".format(attack, i+1, acc_list[i+2]))

    return acc_list


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--arch', type=str, default='dncnn', help='Dataset to train on. One of [dncnn, dae1, dae2].')
    parser.add_argument('--denoiser', type=str, default='mixed+gaussian', help='')
    parser.add_argument('--adv_mode', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--eps', type=float, default=16/256, help='perturbation level')
    parser.add_argument('--norm', type=str, default='inf', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    args = parser.parse_args()

    # load data
    _, test_loader = get_dataloader(args.dataset, args.batch_size, sample_test=False)

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
    net = create_resnet(device=device, grayscale=(args.dataset == 'mnist'))
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
    denoisers = [denoiser]
    test_acc(net, denoisers, test_loader, attack=args.adv_mode, eps=args.eps, norm=args.norm)
