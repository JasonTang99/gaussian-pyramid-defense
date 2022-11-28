#!/bin/bash

# sample command
# python train.py \
#     --dataset mnist \
#     --up_samplers 1 \
#     --down_samplers 1 \
#     --models resnet18 \
#     --pretrained True \
#     --epochs 20 \
#     --batch_size 64 \
#     --lr 1e-3


python train.py \
    --dataset mnist \
    --up_samplers 3 \
    --down_samplers 3 \
    --models resnet18 \
    --pretrained True \
    --epochs 15 \
    --batch_size 64 \
    --lr 1e-5

# 0.8722 0.7595 0.6549

# python train.py \
#     --dataset mnist \
#     --up_samplers 1 \
#     --down_samplers 1 \
#     --models resnet18 \
#     --pretrained True \
#     --epochs 20 \
#     --batch_size 64 \
#     --lr 1e-5
