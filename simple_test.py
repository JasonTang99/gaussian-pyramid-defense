from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#from utils import progress_bar
import os
import argparse

from models.denoisers import DnCNN
from cleverhans.fast_gradient_method import fast_gradient_method
from cleverhans.projected_gradient_descent import projected_gradient_descent
from cleverhans.carlini_wagner_l2 import carlini_wagner_l2
from train_denoiser import img_to_numpy

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_batch(org, adv, denoised, n=6):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow( img_to_numpy(org[i]) )
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

def test_acc(model, denoiser, test_loader, eps, attack):
    model.eval()
    accuracy1 = 0.0
    accuracy2 = 0.0
    accuracy3 = 0.0
    total = 0.0
    
    # with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        if attack == 'fgsm':
            x_adv = fast_gradient_method(net, images, eps, norm=2)
        elif attack == 'pgd':
            x_adv = projected_gradient_descent(net, images, eps, eps_iter=0.01, nb_iter=10, norm=2)
        elif attack == 'cw':
            x_adv = carlini_wagner_l2(net, images, n_classes=10, max_iterations=10)
        # run the model on the test set to predict labels
        outputs = model(images)
        adv_outputs = model(x_adv)
        denoised = denoiser(x_adv)
        denoised_outputs = model(denoised)
        # _, pred = torch.max(outputs.data, 1)
        # _, pred_adv = torch.max(adv_outputs.data, 1)
        # _, pred_dn = torch.max(denoised_outputs.data, 1)
        pred = torch.argmax(outputs, dim=1)
        pred_adv = torch.argmax(adv_outputs, dim=1)
        pred_dn = torch.argmax(denoised_outputs, dim=1)
        total += labels.size(0)
        accuracy1 += (pred == labels).sum().item()
        accuracy2 += (pred_adv == labels).sum().item()
        accuracy3 += (pred_dn == labels).sum().item()
    
    show_batch(images, x_adv, denoised)

    # compute the accuracy over all test images
    accuracy1 = (100 * accuracy1 / total)
    accuracy2 = (100 * accuracy2 / total)
    accuracy3 = (100 * accuracy3 / total)
    print("Test Accuracy no attack: {}".format(accuracy1))
    print("Test Accuracy with {} attack: {}".format(attack, accuracy2))
    print("Test Accuracy with {} attack + {} denoiser: {}".format(attack, args.denoiser, accuracy3))



if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--denoiser', type=str, default='gaussian', help='')
    parser.add_argument('--attack', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--eps', type=float, default=0.1, help='perturbation level')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')

    args = parser.parse_args()

    # load data and split into train and validation sets
    if args.dataset == 'mnist':
        # duplicate to 3 channels
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_data = MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = MNIST(root='./data', train=False, download=True, transform=transform)
        # Model
        model = DnCNN(in_channels=1, out_channels=1, depth=7, hidden_channels=64, use_bias=False).to(device)

    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
        model = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)

    else:
        raise ValueError("Dataset not supported.")

    # prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    net = net.to(device)
    path = os.path.join("trained_models", args.dataset, 'resnet18_0_BL.pth')
    net.load_state_dict(torch.load(path, map_location=device))

    denoiser = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)
    # load pretrained denoiser
    denoiser_path = './trained_denoisers/dncnn_mnist_' + args.denoiser + '.pth'
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))

    test_acc(net, denoiser, test_loader, eps=args.eps, attack=args.attack)
    #test_acc(net, denoiser, test_loader, eps=0.1, attack='pgd')
    #test_acc(net, denoiser, test_loader, eps=0.1, attack='cw')

