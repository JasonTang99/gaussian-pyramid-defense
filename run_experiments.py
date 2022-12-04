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

from tqdm import tqdm

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
        verbose=True
    ):

    # load existing results
    results = {}
    results_path = "attack_results/{}_{}_{}_results".format(
        dataset, attack_type, scaling_factor
    )
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    total_experiments = len(up_down_pairs) * len(interpolations) * len(voting_methods) \
        * len(norms) * len(epsilons) * len(nb_iters) * len(eps_iters) \
        * len(rand_inits) * len(initial_consts)
    pbar = tqdm(total=total_experiments)
    # outer loop defining the model
    for (up_samplers, down_samplers), interpolation, voting_method in itertools.product(
            up_down_pairs, interpolations, voting_methods):
        
        # generate model identifiers
        linear_voting = 'simple_avg' if voting_method == 'simple' else 'weighted_avg'
        nonlinear_voting = 'majority_vote' if voting_method == 'simple' else 'weighted_vote'
        
        linear_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            linear_voting
        )
        voting_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            nonlinear_voting
        )
            
        # Generate attack args
        linear_args = process_args(mode="attack")
        voting_args = process_args(mode="attack")

        if attack_type != "cw":
            if up_samplers >= 2 and scaling_factor == 2:
                batch_size = 16
            # if up_samplers + down_samplers >= 3:
            #     batch_size = 8
        # if verbose:
        print("batch_size: ", batch_size)
        
        for args in [linear_args, voting_args]:
            args.attack_method = attack_type
            args.dataset = dataset
            args.scaling_factor = scaling_factor

            args.up_samplers = up_samplers
            args.down_samplers = down_samplers
            args.interpolation = interpolation
            args.archs = ['resnet18']
        
        linear_args.voting_method = linear_voting
        voting_args.voting_method = nonlinear_voting

        # Post process args
        linear_args = post_process_args(linear_args, mode="attack")
        voting_args = post_process_args(voting_args, mode="attack")

        # Generate models
        linear_model, voting_model = None, None

        # Inner loop defining the attack parameters
        for norm, epsilon, nb_iter, eps_iter, rand_init, initial_const in itertools.product(
                norms, epsilons, nb_iters, eps_iters, rand_inits, initial_consts):
            pbar.update(1)
            if eps_iter > epsilon:
                continue

            for args in [linear_args, voting_args]:
                args.norm = norm
                args.epsilon = epsilon
                args.nb_iter = nb_iter
                args.eps_iter = eps_iter
                args.rand_init = rand_init
                args.initial_const = initial_const

            # Generate identifier for this attack
            linear_attack_id = linear_model_id + "_{}_{}_{}_{}_{}_{}".format(
                norm, epsilon, nb_iter, eps_iter, rand_init, initial_const
            )
            voting_attack_id = voting_model_id + "_{}_{}_{}_{}_{}_{}".format(
                norm, epsilon, nb_iter, eps_iter, rand_init, initial_const
            )

            # Check if attack has already been run
            if linear_attack_id in results and voting_attack_id in results:
                if verbose:
                    print("Attacks {} and {} already run, skipping...".format(
                        linear_attack_id, voting_attack_id
                    ))
                continue
            
            # Load models if not already loaded
            if linear_model is None:
                linear_model = GPEnsemble(linear_args)
                voting_model = GPEnsemble(voting_args)

            # Run attack
            linear_acc, voting_acc = evaluate_attack(linear_args, linear_model, voting_model)

            # Print attack parameters
            if verbose:
                print("=======================================================")
                print(f"Running {attack_type} with parameters:")
                for var in """dataset scaling_factor up_samplers down_samplers interpolation voting_method
                    norm epsilon nb_iter eps_iter rand_init initial_const""".split():
                    print(var, "=", eval(var))
                print(linear_attack_id)
                print(voting_attack_id)

                print(f'Linear Accuracy: {linear_acc}')
                print(f'Voting Accuracy: {voting_acc}')

            # Save results
            results[linear_attack_id] = (
                attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                interpolation, epsilon, nb_iter, eps_iter, rand_init, initial_const,
                linear_voting, linear_acc.detach().cpu().item(), norm
            )
            results[voting_attack_id] = (
                attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                interpolation, epsilon, nb_iter, eps_iter, rand_init, initial_const,
                nonlinear_voting, voting_acc.detach().cpu().item(), norm
            )

            # Store results
            write_results(results_path, results, dictionary=True, overwrite=True)
    pbar.close()

if __name__ == "__main__":
    # run FGSM attack (L2, Linf)
    for scaling in [2.0, 1.1]:
        for dataset in ["mnist", "cifar10"]:
            for norm in [np.inf, 2]:
                if norm == np.inf:
                    epsilons = [x/256 for x in [2, 5, 10, 16]]
                elif norm == 2:
                    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
                
                if scaling == 2.0:
                    up_down_pairs = [
                        *[(0, i) for i in [0, 1, 3]],
                        *[(i, 0) for i in [1, 3]],
                        *[(i, i) for i in [0, 1, 2, 3]]
                    ]
                else:
                    up_down_pairs = [
                        *[(0, i) for i in [3, 5, 7]],
                        *[(i, 0) for i in [3, 5, 7]],
                        *[(i, i) for i in [3, 5, 7]],
                    ]
                interpolations = ["bilinear", "nearest"]
                voting_methods = ['simple', 'weighted']

                print("Running FGSM attack on {} {} {}".format(dataset, scaling, norm))
                
                attack_results = run(
                    attack_type="fgsm",
                    dataset=dataset,
                    scaling_factor=scaling,
                    batch_size=32,
                    up_down_pairs=up_down_pairs,
                    interpolations=interpolations,
                    voting_methods=voting_methods,
                    norms=[norm],
                    epsilons=epsilons,
                    verbose=True
                )

    # run PGD attack (L2, Linf)
    nb_iters = [40]
    rand_inits = [True]

    # general
    scalings, datasets, norms = [2.0, 1.1], ["mnist", "cifar10"], [np.inf, 2]
    # jason gpu?
    scalings, datasets, norms = [1.1], ["mnist", "cifar10"], [np.inf, 2]
    # TODO: steven gpu?
    scalings, datasets, norms = [2.0], ["mnist", "cifar10"], [np.inf, 2]

    for scaling, dataset, norm in itertools.product(scalings, datasets, norms):    
        if norm == np.inf:
            epsilons = [x/256 for x in [2, 5, 10, 16]]
            eps_iters = [5e-4]
        elif norm == 2:
            epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            eps_iters = [0.01]
        
        if scaling == 2.0:
            up_down_pairs = [
                *[(i, i) for i in [0, 2, 3]]
            ]
        else:
            up_down_pairs = [
                *[(i, i) for i in [3, 5, 7]],
            ]
        interpolations = ["bilinear", "nearest"]
        voting_methods = ['simple', 'weighted']
        
        print("Running PGD attack on {} {} {}".format(dataset, scaling, norm))

        attack_results = run(
            attack_type="pgd",
            dataset=dataset,
            scaling_factor=scaling,
            batch_size=64,
            up_down_pairs=up_down_pairs,
            interpolations=interpolations,
            voting_methods=voting_methods,
            norms=[norm],
            epsilons=epsilons,
            eps_iters=eps_iters,
            nb_iters=nb_iters,
            rand_inits=rand_inits,
            verbose=False
        )

    exit(0)

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

