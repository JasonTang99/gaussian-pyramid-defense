#!/bin/bash

# sample train command
python train_ensemble.py \
    --dataset mnist \
    --up_samplers 1 \
    --down_samplers 3 \
    --archs resnet18 \
    --pretrained \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-1 \
    --scaling_factor 2 \
    --interpolation linear

# cifar10
# python train_ensemble.py \
#     --dataset cifar10 \
#     --up_samplers 1 \
#     --down_samplers 3 \
#     --archs resnet18 \
#     --pretrained \
#     --epochs 20 \
#     --batch_size 64 \
#     --lr 1e-1 \
#     --scaling_factor 2 \
#     --interpolation nearest

# smaller
python train_ensemble.py \
    --dataset mnist \
    --up_samplers 7 \
    --down_samplers 7 \
    --archs resnet18 \
    --pretrained \
    --epochs 20 \
    --batch_size 64 \
    --lr 5e-2 \
    --scaling_factor 1.1 \
    --interpolation nearest

# cifar10
# python train_ensemble.py \
#     --dataset cifar10 \
#     --up_samplers 0 \
#     --down_samplers 7 \
#     --archs resnet18 \
#     --pretrained \
#     --epochs 20 \
#     --batch_size 64 \
#     --lr 5e-2 \
#     --scaling_factor 1.1 \
#     --interpolation nearest
