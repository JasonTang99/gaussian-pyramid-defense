#!/bin/bash

# sample train command
python train_ensemble.py \
    --dataset mnist \
    --up_samplers 2 \
    --down_samplers 3 \
    --archs resnet18 \
    --pretrained \
    --epochs 12 \
    --batch_size 64 \
    --lr 5e-5 \
    --scaling_factor 2
