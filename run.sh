#!/bin/bash

# sample train command
python train_ensemble.py \
    --dataset mnist \
    --up_samplers 2 \
    --down_samplers 3 \
    --archs resnet18 \
    --pretrained \
    --epochs 15 \
    --batch_size 64 \
    --lr 1e-5 \
    --scaling_factor 2
