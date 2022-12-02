import os
import numpy as np
import argparse
import itertools
from torchvision.transforms import InterpolationMode
import pickle

from parse_args import process_args, post_process_args
from attack import evaluate_attack
from models.gp_ensemble import GPEnsemble
from utils import read_results, write_results

def run(attack_type,
        dataset,
        scaling_factor,
        batch_size,
        up_down_pairs,
        interpolations,
        voting_methods,
        norms,
        epsilons=[0.1],         # fgsm, pgd only
        nb_iters=[10],          # pgd only
        eps_iters=[0.01],       # pgd only
        rand_inits=[False],     # pgd only
        initial_consts=[0.1],   # cw only
    ):

    # load existing results
    results = {}
    results_path = "attack_results/{}_{}_{}_results".format(
        dataset, attack_type, scaling_factor
    )
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(len(results))
    
    # outer loop defining the model
    for (up_samplers, down_samplers), interpolation, voting_method in itertools.product(
            up_down_pairs, interpolations, voting_methods):
        # generate model identifier
        model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            voting_method,
        )
        
        # Generate default attack args
        args = process_args(mode="attack")
        # Modify model args
        args.attack_method = attack_type
        args.dataset = dataset
        args.scaling_factor = scaling_factor
        
        args.batch_size = batch_size
        if attack_type != "cw":
            if up_samplers >= 2 and scaling_factor == 2:
                args.batch_size = 32
            if up_samplers + down_samplers >= 3:
                args.batch_size = 16
        print("batch_size: {}".format(args.batch_size))
        
        args.up_samplers = up_samplers
        args.down_samplers = down_samplers
        args.interpolation = interpolation
        args.voting_method = voting_method

        args.archs = ['resnet18']

        # Post-process args
        args = post_process_args(args, mode="attack")

        # Generate Model
        model = GPEnsemble(args)

        # Inner loop defining the attack parameters
        for norm, epsilon, nb_iter, eps_iter, rand_init, initial_const in itertools.product(
                norms, epsilons, nb_iters, eps_iters, rand_inits, initial_consts):

            args.norm = norm
            args.epsilon = epsilon
            args.nb_iter = nb_iter
            args.eps_iter = eps_iter
            args.rand_init = rand_init
            args.initial_const = initial_const

            # Generate identifier for this attack
            attack_id = model_id + "_{}_{}_{}_{}_{}_{}".format(
                norm, epsilon, nb_iter, eps_iter, rand_init, initial_const
            )

            # Check if attack has already been run
            if attack_id in results:
                print(f"Attack {attack_id} already run, skipping...")
                continue

            # Print attack parameters
            print("=======================================================")
            print(f"Running {attack_type} with parameters:")
            for var in """dataset scaling_factor up_samplers down_samplers interpolation voting_method
                norm epsilon nb_iter eps_iter rand_init initial_const""".split():
                print(var, "=", eval(var))

            # Run attack
            test_acc, norm = evaluate_attack(args, model)

            # Save results
            results[attack_id] = (
                attack_type,
                dataset,
                scaling_factor,
                up_samplers,
                down_samplers,
                interpolation,
                voting_method,
                epsilon,
                nb_iter,
                eps_iter,
                rand_init,
                initial_const,
                # Results
                test_acc.detach().cpu().item(),
                norm,
            )

            # Store results
            write_results(results_path, results, dictionary=True, overwrite=True)

if __name__ == "__main__":
    # run FGSM attack (L1, L2, Linf)
    for dataset in ["mnist", "cifar10"]:
        for norm in [np.inf, 1, 2]:
            if norm == 1:
                epsilons = [x/256 for x in [2, 5, 10, 16]]
            elif norm == 2:
                epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                epsilons = [0.01, 0.1, 0.25]
            up_down_pairs = [
                *[(0, i) for i in range(0, 4)],
                *[(i, 0) for i in range(1, 4)],
                (1, 1), (2, 2), (3, 3),
            ]
            interpolations = ["bilinear"] #, "nearest"]
            voting_methods = [
                'simple_avg',
                'weighted_avg',
                # 'majority_vote',
                # 'weighted_vote',
            ]
            
            attack_results = run(
                attack_type="fgsm",
                dataset=dataset,
                scaling_factor=2.0,
                batch_size=128,
                up_down_pairs=up_down_pairs,
                interpolations=interpolations,
                voting_methods=voting_methods,
                norms=[norm],
                epsilons=epsilons,
            )

    exit(0)

    # run PGD attack (L2, Linf)
    eps_iters = [0.005, 0.01]

    # run CW attack (L2)
    up_down_pairs = [
        (0, 0),
        # *[(0, i) for i in range(0, 4)],
        # *[(i, 0) for i in range(1, 4)],
        # (1, 1), 
        # (2, 2), 
        (3, 3),
    ]
    interpolations = ["bilinear"] #, "nearest"]
    voting_methods = [
        'simple_avg',
        'weighted_avg',
        # 'majority_vote',
        # 'weighted_vote',
    ]
    norms = [2] # [1, 2, np.inf]
    initial_consts = [1e-8, 1e-6] #, 1e-4, 1e-2]
    
    attack_results = run(
        attack_type="cw",
        dataset='mnist',
        scaling_factor=2.0,
        batch_size=128,
        up_down_pairs=up_down_pairs,
        interpolations=interpolations,
        voting_methods=voting_methods,
        norms=norms,
        # epsilons=epsilons,      # fgsm, pgd only
        # nb_iters=[10],          # pgd only
        # eps_iters=[0.01],       # pgd only
        # rand_inits=[False],     # pgd only
        initial_consts=initial_consts,   # cw only
    )

