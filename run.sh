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
    --up_samplers 1 \
    --down_samplers 1 \
    --models resnet18 \
    --pretrained True \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4

python train.py \
    --dataset mnist \
    --up_samplers 1 \
    --down_samplers 1 \
    --models resnet18 \
    --pretrained True \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-5
