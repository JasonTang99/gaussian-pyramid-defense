from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from models.denoisers import DnCNN
from cleverhans.fast_gradient_method import fast_gradient_method
from cleverhans.projected_gradient_descent import projected_gradient_descent
from cleverhans.carlini_wagner_l2 import carlini_wagner_l2


# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on: ", device)


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def add_noise(x, mode, noise_level):
    if mode == 'gaussian':
        noisy = x + torch.randn_like(x) * noise_level
    elif mode == 'fgsm':
        noisy = fast_gradient_method(net, x, eps=noise_level, norm=np.inf)
    elif mode == 'pgd':
        noisy = projected_gradient_descent(net, x, eps=noise_level, eps_iter=0.01, nb_iter=20, norm=np.inf)
    else: pass

    return noisy

def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0., 1.)

def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out

def show_batch(model, test_loader, mode, n=6):
    # obtain one batch of training images
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    # add some noise
    noisy_imgs = add_noise(images, mode, noise_level=0.1)
    denoised = model(noisy_imgs)

    plt.figure(figsize=(20, 6))

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        img = img_to_numpy(images[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i +1 + n)
        img = img_to_numpy(noisy_imgs[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display denoised image
        ax = plt.subplot(3, n, i +1 + n + n)
        img = img_to_numpy(denoised[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.65, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.35, "DENOISED RECONSTRUCTED IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace = 0.5 )
    plt.show()


def evaluate_model(model, data_loader, mode, noise_level, test=False):
    if test: model.eval()

    avg_psnr=0
    avg_ssim=0
    total=0
    for data in tqdm(data_loader):
        images, _ = data
        images = images.to(device)
        noisy_imgs = add_noise(images, mode, noise_level)
        #noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        output = model(noisy_imgs)
        #output = output.view(len(images), 3, 32, 32)
        #output = output.detach().cpu()
        for i in range(len(images)):
            original = img_to_numpy(images[i])
            denoised = img_to_numpy(output[i])
            avg_psnr += PSNR(original, denoised)
            avg_ssim += SSIM(original, denoised, multichannel=True)

        total += len(images)

    print("\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr/total, avg_ssim/total))


def train(args, model, model_name, train_loader, val_loader):

    # get args
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    noise_range = args.noise_level
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #lambda1 = lambda x: 0.65 ** x
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    model.train()

    pbar = tqdm(total=len(train_data) * num_epochs // batch_size)
    for epoch in range(1, num_epochs + 1):

        train_loss = 0.0
        for idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # generate random noise level within given range
            noise_level = np.random.uniform(noise_range[0], noise_range[-1])
            noisy_imgs = add_noise(images, args.adv_mode, noise_level)
                    
            optimizer.zero_grad()
            # denoiser
            denoised = model(noisy_imgs)

            loss = criterion(denoised, images)
            loss.backward()
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

            if not idx % 200:
                psnr = calc_psnr(denoised, images)
                baseline_psnr = calc_psnr(noisy_imgs, images)
                print("\nTraining Loss: {:.4f} | Baseline PSNR: {:.2f} | PSNR: {:.2f}".format(train_loss/(idx + 1), baseline_psnr, psnr))

            pbar.update(1)
        
        #scheduler.step()
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('\nEpoch: {} | Training Loss: {:.4f}'.format(epoch, train_loss))
        # evaluate model on validation set
        # evaluate_model(model, val_loader, args.adv_mode, noise_level=0.1) 

    pbar.close()

    # save model
    if not os.path.isdir('trained_denoisers'):
        os.mkdir('trained_denoisers')

    torch.save(model.state_dict(), os.path.join("trained_denoisers", model_name))


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    #parser.add_argument('--pretrained', type=bool, default=True, help='Whether to start from pretrained ensemble models.')
    parser.add_argument('--adv_mode', type=str, default='gaussian', help='type of adversarial noise')
    parser.add_argument('--noise_level', type=float, nargs='+', default=[0.05, 0.2], help='noise level range, sigma for gaussian, eps for adversarial')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

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
        train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])
    else:
        raise ValueError("Dataset not supported.")

    # prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    #print("Number of images in train set: ", len(train_loader)*args.batch_size)
    #print("Number of images in val set: ", len(val_loader)*args.batch_size)

    # denoiser model
    model = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)
    
    # classification model
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    net = net.to(device)
    # load from pretrained
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_0_BL.pth'), map_location=device))

    # noise_range = args.noise_level
    # if len(noise_range) < 2: # fixed noise level
    #     model_name = f"dncnn_{args.dataset}_{args.adv_mode}_{noise_range[0]}.pth"
    # else:
    #     model_name = f"dncnn_{args.dataset}_{args.adv_mode}_{noise_range[0]}_{noise_range[-1]}.pth"
    model_name = f"dncnn_{args.dataset}_{args.adv_mode}.pth"
    print(model_name)

    if os.path.exists(os.path.join("trained_denoisers", model_name)):
        print("Model already exists. Skip Training.")
        model.load_state_dict(torch.load(os.path.join("trained_denoisers", model_name), map_location=device))
    else:
        train(args, model, model_name, train_loader, val_loader)

    # display from a batch of images
    show_batch(model, train_loader, args.adv_mode)
    evaluate_model(model, test_loader, mode=args.adv_mode, noise_level=0.1, test=True)
