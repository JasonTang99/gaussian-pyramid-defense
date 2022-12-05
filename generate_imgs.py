from load_ens_adv_train import get_ens_adv_model
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
from cleverhans_fixed.carlini_wagner_l2 import carlini_wagner_l2

# load ensemble
args = process_args(mode="attack")
args.dataset = "cifar10"
args.up_samplers = 0
args.down_samplers = 0
# args.up_samplers = 3
# args.down_samplers = 3
args.scaling_factor = 2.0
args.interpolation = "bilinear"
args = post_process_args(args, mode="attack")
gp_model = GPEnsemble(args)
gp_model.eval()

# load model
advens_model = get_ens_adv_model()
advens_model.eval()

# to args.device
gp_model = gp_model.to(args.device)
advens_model = advens_model.to(args.device)


# load data
test_loader = load_data(args, train=False)
# get first image
img, label = next(iter(test_loader))
i = 3
img = img[i].unsqueeze(0).to(args.device)
label = label[i].unsqueeze(0).to(args.device)
print(img.shape, label)

imgs = [img]
# generate adversarial images
# fgsm
imgs.append(fast_gradient_method(
    model_fn=gp_model,
    x=img,
    eps=16.0/256,
    norm=np.inf,
    clip_min=0.0,
    clip_max=1.0,
))
# pgd
# imgs.append(projected_gradient_descent(
#     model_fn=gp_model,
#     x=img,
#     eps=16.0/256,
#     eps_iter=5e-4,
#     nb_iter=40,
#     norm=np.inf,
#     clip_min=0.0,
#     clip_max=1.0,
#     rand_init=True,
#     sanity_checks=False
# ))
# # cwl2
# imgs.append(carlini_wagner_l2(
#     model_fn=gp_model,
#     x=img,
#     n_classes=10,
#     lr=5e-3,
#     binary_search_steps=10,
#     max_iterations=100,
#     initial_const=1e-8,
# ))

# calculate differences from original image
diffs = [adv_img - img for adv_img in imgs]

# calculate predictions percentages after softmax
softmax = torch.nn.Softmax(dim=1)
preds = [softmax(gp_model(img)) for img in imgs]
percentages = [pred.max(dim=1)[0] for pred in preds]
labels = [pred.max(dim=1)[1] for pred in preds]
print(preds)
print(percentages)
print(labels)
# detach and move to numpy
imgs = [img.detach().cpu().numpy() for img in imgs]
diffs = [diff.detach().cpu().numpy() for diff in diffs]

# plot images
fig, axs = plt.subplots(2, 4)
for i in range(2):
    print(imgs[i].shape, diffs[i].shape)
    axs[0, i].imshow(imgs[i][0].transpose(1, 2, 0))
    axs[1, i].imshow(diffs[i][0].transpose(1, 2, 0))
    # caption with prediction
    axs[0, i].set_title(f"pred: {preds[i]}")
plt.show()



# plot image
# plt.imshow(img[0].permute(1, 2, 0))
# plt.show()