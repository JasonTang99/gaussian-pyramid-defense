import os
import numpy as np
import argparse
import itertools
import torch
from torchvision.transforms import InterpolationMode
import pickle

from parse_args import process_args, post_process_args
from attack import evaluate_attack, evaluate_cw_l2
from models.gp_ensemble import GPEnsemble
from utils import read_results, write_results
from load_ens_adv_train import get_ens_adv_model

from tqdm import tqdm

def run(attack_type,
        dataset,
        batch_size,
        norms,
        epsilons=[0.1],         # fgsm, pgd only
        nb_iters=[10],          # pgd only
        eps_iters=[0.01],       # pgd only
        rand_inits=[False],     # pgd only
    ):

    # load existing results
    results = {}
    results_path = "attack_results/advens_results"
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    # defining the model
    model = get_ens_adv_model()
    model.eval()

    # Generate attack args
    args = process_args(mode="attack")
    args.attack_method = attack_type
    args.dataset = dataset

    args = post_process_args(args, mode="attack")

    # Loop defining the attack parameters
    for norm, epsilon, nb_iter, eps_iter, rand_init in itertools.product(
            norms, epsilons, nb_iters, eps_iters, rand_inits):
        if eps_iter > epsilon:
            continue

        args.norm = norm
        args.epsilon = epsilon
        args.nb_iter = nb_iter
        args.eps_iter = eps_iter
        args.rand_init = rand_init

        # Generate identifier for this attack
        attack_id = "advens_{}_{}_{}_{}_{}_{}_{}".format(
            attack_type, dataset, norm, epsilon, nb_iter, eps_iter, rand_init
        )

        # Check if attack has already been run
        if attack_id in results:
            if verbose:
                print("Attack {} already run".format(attack_id))
            continue
        
        # Run attack
        test_acc, _ = evaluate_attack(args, model)

        # Save results
        results[attack_id] = (
            attack_type, dataset, epsilon, nb_iter, eps_iter, rand_init,
            test_acc.detach().cpu().item(), norm
        )

        # Store results
        write_results(results_path, results, dictionary=True, overwrite=True)

def run_cw(
        dataset,
        batch_size,
        epsilons,
        initial_consts=[1e-8],   # cw only
        verbose=True
    ):
    attack_type = "cw"
    norms = [2.0]

    # load existing results
    results = {}
    results_path = "attack_results/advens_results"
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
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
        for norm, initial_const in itertools.product(norms, initial_consts):
            pbar.update(1)

            for args in [linear_args, voting_args]:
                args.norm = norm
                args.initial_const = initial_const

            # Generate identifier for this attack
            attack_id = linear_model_id + "_{}_{}_{}".format(
                norm, initial_const, epsilons
            )
            # Check if attack has already been run
            if attack_id in results:
                continue
            
            # Load models if not already loaded
            if linear_model is None:
                linear_model = GPEnsemble(linear_args)
                voting_model = GPEnsemble(voting_args)

            # Run attack
            linear_acc, voting_acc = evaluate_cw_l2(linear_args, 
                linear_model, voting_model, epsilons=epsilons)

            for epsilon, lacc, vacc in zip(epsilons, linear_acc, voting_acc):
                linear_attack_id = linear_model_id + "_{}_{}_{}".format(
                    norm, epsilon, initial_const
                )
                voting_attack_id = voting_model_id + "_{}_{}_{}".format(
                    norm, epsilon, initial_const
                )
                # print(linear_attack_id)
                # print(voting_attack_id)

                # Save results
                results[linear_attack_id] = (
                    attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                    interpolation, epsilon, initial_const,
                    linear_voting, lacc.detach().cpu().item(), norm
                )
                results[voting_attack_id] = (
                    attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                    interpolation, epsilon, initial_const,
                    nonlinear_voting, vacc.detach().cpu().item(), norm
                )

            # Store results
            write_results(results_path, results, dictionary=True, overwrite=True)
    pbar.close()

def experiment_fgsm():
    norm = np.inf
    epsilons = [x/256 for x in [2, 5, 10, 16]]
    interpolations = ["bilinear"]
    voting_methods = ['simple', 'weighted']

    for scaling in [2.0, 1.1]:
        for dataset in ["mnist", "cifar10"]:
            if scaling == 2.0:
                up_down_pairs = [
                    *[(0, i) for i in range(0, 4)],
                    *[(i, 0) for i in range(1, 4)],
                    *[(i, i) for i in [0, 1, 2, 3]]
                ]
            else:
                up_down_pairs = [
                    *[(0, i) for i in [3, 5, 7]],
                    *[(i, 0) for i in [3, 5, 7]],
                    *[(i, i) for i in [3, 5, 7]],
                ]

            print("Running FGSM attack on {} {} {}".format(dataset, scaling, norm))
            
            attack_results = run(
                attack_type="fgsm",
                dataset=dataset,
                scaling_factor=scaling,
                batch_size=64,
                up_down_pairs=up_down_pairs,
                interpolations=interpolations,
                voting_methods=voting_methods,
                norms=[norm],
                epsilons=epsilons,
                verbose=True
            )

def experiment_pgd():
    # run PGD attack (Linf)
    nb_iters = [40]
    rand_inits = [True]
    norm = np.inf
    epsilons = [x/256 for x in [2, 5, 10, 16]]
    eps_iters = [5e-4]
    interpolations = ["bilinear"]
    voting_methods = ['simple', 'weighted']
    
    scalings, datasets = [2.0, 1.1], ["mnist", "cifar10"]

    for scaling, dataset in itertools.product(scalings, datasets):    
        if scaling == 2.0:
            up_down_pairs = [
                *[(i, i) for i in [0, 3]]
            ]
        else:
            up_down_pairs = [
                *[(i, i) for i in [3, 5, 7]],
            ]

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

def experiment_cw():
    # run CW attack (L2)
    norm = 2.0
    epsilons = [0.5, 1.0, 2.0, 3.5]
    interpolations = ["bilinear"]
    voting_methods = ['simple', 'weighted']
    
    # Jason
    scalings, datasets = [1.1], ["mnist"]
    # Steven
    # scalings, datasets = [2.0], ["mnist"]

    for scaling, dataset in itertools.product(scalings, datasets):    
        if scaling == 2.0:
            up_down_pairs = [
                *[(i, i) for i in [0, 3]]
            ]
        else:
            up_down_pairs = [
                *[(i, i) for i in [5, 7]],
            ]

        print("Running CW attack on {} {} {}".format(dataset, scaling, norm))

        attack_results = run_cw(
            dataset=dataset,
            scaling_factor=scaling,
            batch_size=64,
            up_down_pairs=up_down_pairs,
            voting_methods=voting_methods,
            epsilons=epsilons,
            verbose=False
        )

if __name__ == "__main__":
    # experiment_fgsm()
    # experiment_pgd()
    experiment_cw()


