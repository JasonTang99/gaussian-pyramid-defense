#!/bin/bash

# sample command
python train.py \
    --dataset mnist \
    --up_samplers 0 \
    --down_samplers 1 \
    --models resnet18 \
    --pretrained True \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3
