import os
import numpy as np
import argparse
import itertools
from torchvision.transforms import InterpolationMode
import pickle

from parse_args import parse_args
from attack import run_one_attack

def run(
        attack_type,
        dataset,
        epsilons,
        up_down_pairs,
        interpolations,
        voting_methods,
    ):

    results = []
    for epsilon, (up_samplers, down_samplers), interpolation, voting_method \
        in itertools.product(
            epsilons,
            up_down_pairs,
            interpolations,
            voting_methods,
        ):
            print("=======================================================")
            print("Running attack with parameters:")
            print(f"attack_type: {attack_type}")
            print(f"dataset: {dataset}")
            print(f"epsilon: {epsilon}")
            print(f"up_samplers: {up_samplers}")
            print(f"down_samplers: {down_samplers}")
            print(f"interpolation: {interpolation}")
            print(f"voting_method: {voting_method}")
            print("=======================================================")

            # Initialize args
            args = parse_args(mode="attack", verbose=False)
            args.dataset = dataset
            args.up_samplers = up_samplers
            args.down_samplers = down_samplers
            args.interpolation = interpolation
            args.scale_factor = 2.0
            args.archs = ['resnet18'] * (up_samplers + down_samplers + 1)
            args.batch_size = 64

            args.voting_method = voting_method
            args.attack_method = attack_type
            args.norm = 2
            
            args.epsilon = epsilon
            
            # TODO add other args later
            # print("Arguments:")
            # print('  ' + '\n  '.join(f'{k:15}= {v}' for k, v in vars(args).items()))

            # Run attack
            test_acc = run_one_attack(args)
            results.append((
                attack_type,
                dataset,
                epsilon,
                up_samplers,
                down_samplers,
                interpolation,
                voting_method,
                test_acc,
            ))
    return results



            

    # os.system("""python attack.py \
    #     --dataset mnist \
    #     --up_samplers 2 \
    #     --down_samplers 0 \
    #     --archs resnet18 \
    #     --batch_size 64 \
    #     --scaling_factor 2 \
    #     --attack_method fgsm \
    #     --epsilon 1.5 \
    #     --norm 2"""
    # )


if __name__ == "__main__":
    # run FGSM attack (L1, L2, Linf)
    epsilons = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    up_down_pairs = [
        *[(0, i) for i in range(0, 4)],
        *[(i, 0) for i in range(1, 3)],
        (2, 3)
    ]
    interpolations = [
        # InterpolationMode.NEAREST,
        InterpolationMode.BILINEAR,
    ]
    voting_methods = [
        'simple_avg',
        # 'weighted_avg',
        # 'majority_vote',
        # 'weighted_vote',
    ]
    
    attack_results = run(
        dataset='mnist',
        attack_type="fgsm",
        epsilons=epsilons,
        up_down_pairs=up_down_pairs,
        interpolations=interpolations,
        voting_methods=voting_methods,
    )

    # Save results
    with open('attack_results.pkl', 'wb') as f:
        pickle.dump(attack_results, f)

    # run PGD attack (L2, Linf)
    eps_iters = [0.005, 0.01]
    
