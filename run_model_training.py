import numpy as np
import os
import random

def run(mode="gp"):
    ### Code to train GPEnsemble ###
    if mode == "gp":
        for dataset in ['mnist', 'cifar10']:
            for interp in ['nearest', 'bilinear']:
                for (up, down, epochs, scale, bs) in [
                    (0, 3, 20, 2.0, 64), (3, 0, 12, 2.0, 16), 
                    (0, 7, 15, 1.1, 64), (7, 0, 12, 1.1, 32)]:

                    cmd = f"""python train_ensemble.py \
                        --dataset {dataset} \
                        --up_samplers {up} \
                        --down_samplers {down} \
                        --archs resnet18 \
                        --pretrained \
                        --epochs {epochs} \
                        --batch_size {bs} \
                        --lr 5e-2 \
                        --scaling_factor {scale} \
                        --interpolation {interp}"""
                    os.system(cmd)
    
    ### Code to train baselines ###
    else:
        epochs_list = list(range(40, 50))
        batch_size_list = [24, 32, 48, 64]
        lr_list = list(np.arange(0.005, 0.05, 0.005))
        for dataset in ['cifar10']:
            for i in range(9):
                fp = f"trained_models/{dataset}/baseline_{i}.pth"
                if os.path.exists(fp):
                    print(f"Already trained {dataset} baseline {i}, skipping")
                    continue
                # Randomly choose hyperparameters
                epochs = random.choice(epochs_list)
                batch_size = random.choice(batch_size_list)
                lr = random.choice(lr_list)

                os.system(f"""python train_ensemble.py \
                    --dataset {dataset} \
                    --up_samplers 0 \
                    --down_samplers 0 \
                    --archs resnet18 \
                    --pretrained \
                    --epochs {epochs} \
                    --batch_size {batch_size} \
                    --lr {lr} \
                    --scaling_factor 2""")
                os.system(f"mv trained_models/{dataset}/resnet18_2.0+0_BL.pth {fp}")

                print(f"epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")

if __name__ == "__main__":
    run(mode="gp")
    run(mode="baseline")