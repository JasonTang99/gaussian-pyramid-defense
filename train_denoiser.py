import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from models.denoisers import DnCNN, REDNet20
from custom_dataset import AdversarialDataset, get_dataloader, img_to_numpy
from utils import create_resnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on: ", device)

# def add_noise(x, mode, noise_level):
#     if mode == 'gaussian':
#         noisy = x + torch.randn_like(x) * noise_level
#     elif mode == 'fgsm':
#         noisy = fast_gradient_method(net, x, eps=noise_level, norm=np.inf)
#     elif mode == 'pgd':
#         noisy = projected_gradient_descent(net, x, eps=noise_level, eps_iter=0.01, nb_iter=20, norm=np.inf)
#     else: pass
#     return noisy

def add_gaussian_noise(x, noise_level):
    noisy = x + torch.randn_like(x) * noise_level
    return noisy

def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out

def show_batch(images, noisy, denoised, n=6):
    # # obtain one batch of training images
    # dataiter = iter(test_loader)
    # images, labels = next(dataiter)
    # images, labels = images.to(device), labels.to(device)
    # # add some noise
    # if mode == "gaussian":
    #     nosiy = add_gaussian_noise(images, noise_level=0.1)
    # else:
    #     nosiy = labels

    # denoised = model(nosiy)

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
        img = img_to_numpy(noisy[i])
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


def evaluate_model(model, data_loader, mode, test=False):
    if test: model.eval()

    avg_psnr=0
    avg_ssim=0
    total=0
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        # add some noise
        if mode == "gaussian":
            noisy = add_gaussian_noise(images, noise_level=0.1)
        else:
            noisy = labels
        #noisy = add_noise(images, mode, noise_level)
        output = model(noisy)
        #output = output.view(len(images), 3, 32, 32)
        #output = output.detach().cpu()
        for i in range(len(images)):
            original = img_to_numpy(images[i])
            denoised = img_to_numpy(output[i])
            avg_psnr += PSNR(original, denoised)
            avg_ssim += SSIM(original, denoised, multichannel=True)

        total += len(images)
    
    show_batch(images, noisy, output, n=10)

    print("\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr/total, avg_ssim/total))


def train(args, model, train_loader, val_loader):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    pbar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(1, args.epochs + 1):

        train_loss = 0.0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if args.adv_mode == 'gaussian':
                noise_level = np.random.uniform(args.noise_level[0], args.noise_level[-1])
                noisy_imgs = add_gaussian_noise(images, noise_level)
            else: # attack
                noisy_imgs = labels
            # noisy_imgs = add_noise(images, args.adv_mode, noise_level)
                    
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
        
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('\nEpoch: {} | Training Loss: {:.4f}'.format(epoch, train_loss))
        # evaluate model on validation set
        # evaluate_model(model, val_loader, args.adv_mode, test=False) 

    show_batch(images, noisy_imgs, denoised, n=10)
    pbar.close()


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='dncnn', help='Dataset to train on. One of [dncnn, dae].')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    #parser.add_argument('--pretrained', type=bool, default=True, help='Whether to start from pretrained ensemble models.')
    parser.add_argument('--adv_mode', type=str, default='gaussian', help='type of adversarial noise')
    parser.add_argument('--norm', type=str, default='2', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--noise_level', type=float, nargs='+', default=[0.05, 0.2], help='noise level range, sigma for gaussian, eps for adversarial')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    args = parser.parse_args()

    if args.adv_mode == "gaussian":
        train_loader, test_loader = get_dataloader(args, val=False)
    else: 
        # get data from pre-generated adversarial examples
        train_data = AdversarialDataset(args, train=True)
        test_data = AdversarialDataset(args, train=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print("Train Size: ", len(train_loader.dataset))

    # denoiser model
    if args.arch == 'dncnn':
        model = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)
    elif args.arch == 'dae':
        model = REDNet20(in_channels=3, out_channels=3, use_bias=False).to(device)

    model_name = f"{args.arch}_{args.dataset}_{args.adv_mode}.pth"
    print(model_name)
    
    # classification model
    net = create_resnet(device=device)
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_2.0+0_BL.pth'), map_location=device))


    if os.path.exists(os.path.join("trained_denoisers", model_name)):
        print("Model already exists. Skip Training.")
        model.load_state_dict(torch.load(os.path.join("trained_denoisers", model_name), map_location=device))
    else:
        print("=================== Start Training ====================")
        print(f"denoiser: {args.arch}")
        print(f"dataset: {args.dataset}")
        print(f"noise type: {args.adv_mode}")
        print(f"noise level: {args.noise_level}")
        print(f"learning rate: {args.lr}")
        print(f"batch size: {args.batch_size}")
        print("=======================================================")
        train(args, model, train_loader, test_loader)
        # save model
        if not os.path.isdir('trained_denoisers'):
            os.mkdir('trained_denoisers')

        torch.save(model.state_dict(), os.path.join("trained_denoisers", model_name))

    # eval
    evaluate_model(model, test_loader, args.adv_mode, test=True)
