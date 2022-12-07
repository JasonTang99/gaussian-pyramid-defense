from load_model import load_advens_model
from datasets import load_data
from parse_args import process_args, post_process_args
from models.gp_ensemble import GPEnsemble
import numpy as np
import torch

import matplotlib.pyplot as plt
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

def get_attack(args, model, img, label):
    imgs = [img]
    # fgsm
    imgs.append(fast_gradient_method(
        model_fn=model,
        x=img,
        eps=6.0/256,
        norm=np.inf,
        clip_min=0.0,
        clip_max=1.0,
    ))
    # pgd
    imgs.append(projected_gradient_descent(
        model_fn=gp_model,
        x=img,
        eps=6.0/256,
        eps_iter=5e-4,
        nb_iter=40,
        norm=np.inf,
        clip_min=0.0,
        clip_max=1.0,
        rand_init=True,
        sanity_checks=False
    ))
    # cwl2
    imgs.append(carlini_wagner_l2(
        model_fn=gp_model,
        x=img,
        n_classes=10,
        lr=1e-2,
        binary_search_steps=10,
        max_iterations=100,
        initial_const=1e-4,
    ))

    # calculate differences from original image
    diffs = [torch.abs(adv_img - img) for adv_img in imgs]
    print([i.max() for i in diffs])
    # calculate l2 and linf norms
    l2s = [torch.norm(diff, p=2).detach().item() for diff in diffs]
    linfs = [torch.norm(diff, p=float('inf')).detach().item() for diff in diffs]

    # calculate predictions percentages after softmax
    print(label)
    softmax = torch.nn.Softmax(dim=1)
    preds = [softmax(gp_model(img)) for img in imgs]
    percentages = [pred.max(dim=1)[0].detach().item() for pred in preds]
    adv_labels = [cifar10_classes[pred.max(dim=1)[1].detach().item()] for pred in preds]
    true_percentages = [pred[0][label.item()].detach().item() for pred in preds]
    print(preds)
    print(percentages)
    print(adv_labels)
    print(true_percentages)
    print(l2s)
    print(linfs)
    # detach and move to numpy
    imgs = [img.detach().cpu().numpy() for img in imgs]
    diffs = [diff.detach().cpu().numpy() for diff in diffs]
    
    return imgs, diffs, true_percentages, adv_labels, percentages, l2s, linfs


if __name__ == "__main__":
    # load ensemble
    args = process_args(mode="attack")
    args.dataset = "cifar10"
    # args.up_samplers = 0
    # args.down_samplers = 0
    args.up_samplers = 3
    args.down_samplers = 3
    args.scaling_factor = 2.0
    args.interpolation = "bilinear"
    args = post_process_args(args, mode="attack")
    gp_model = GPEnsemble(args)
    gp_model.eval()

    # load model
    advens_model = load_advens_model()
    advens_model.eval()

    # to args.device
    gp_model = gp_model.to(args.device)
    advens_model = advens_model.to(args.device)

    # load data
    test_loader = load_data(args, train=False)
    # get first image
    img, label = next(iter(test_loader))
    image_idx = 4
    img = img[image_idx].unsqueeze(0).to(args.device)
    label = label[image_idx].unsqueeze(0).to(args.device)

    # get attacks
    gp_imgs, gp_diffs, gp_true_percentages, gp_adv_labels, gp_percentages, gp_l2s, gp_linfs = \
        get_attack(args, gp_model, img, label)
    advens_imgs, advens_diffs, advens_true_percentages, advens_adv_labels, advens_percentages, advens_l2s, advens_linfs = \
        get_attack(args, advens_model, img, label)
    
    # save img
    # img = gp_imgs[1]
    # img = img.transpose(1, 2, 0)
    # img = (img * 255).astype(np.uint8)
    # from PIL import Image
    # img = Image.fromarray(img)
    # img.save("img.png")

    
    # plot images
    fig, axs = plt.subplots(2, 4)
    for i in range(4):
        # Label with prediction and percentage
        if i == 0:
            axs[0, i].set_title(f"{gp_adv_labels[i]} {gp_percentages[i]*100:.2f}%", fontsize=11)
            axs[0, i].imshow(gp_imgs[i][0].transpose(1, 2, 0))
            axs[1, i].set_title(f"{advens_adv_labels[i]} {advens_true_percentages[i]*100:.2f}%", fontsize=11)
            axs[1, i].imshow(gp_imgs[i][0].transpose(1, 2, 0))
            continue

        axs[0, i].set_title(f"{gp_adv_labels[i]} {gp_percentages[i]*100:.2f}%\n {cifar10_classes[label]} {gp_true_percentages[i]*100:.2f}%\nLinf: {gp_linfs[i]:.3f}\n L2: {gp_l2s[i]:.3f}", fontsize=11)
        axs[0, i].imshow(gp_imgs[i][0].transpose(1, 2, 0))

        axs[1, i].set_title(f"{advens_adv_labels[i]} {advens_percentages[i]*100:.2f}%\n {cifar10_classes[label]} {advens_true_percentages[i]*100:.2f}%\nLinf: {advens_linfs[i]:.3f}\n L2: {advens_l2s[i]:.3f}", fontsize=11)
        axs[1, i].imshow(advens_imgs[i][0].transpose(1, 2, 0))
    
    # for ax in axs.flat:
    #     ax.axis('off')
    for i, title in enumerate(["GPEnsemble", "AdvEns"]):
        axs[i, 0].set_ylabel(title, fontsize=12, rotation=90, labelpad=0)
    # turn off ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # make plot tight
    plt.margins(0,0)
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.savefig("graphs/cifar10_attack.png", bbox_inches='tight')
    plt.show()
