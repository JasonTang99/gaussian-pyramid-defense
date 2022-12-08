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
from load_model import load_advens_model, load_fastadv_model

from tqdm import tqdm

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(model,
        model_name,
        attack_type,
        dataset,
        batch_size,
        norms=[np.inf],
        epsilons=[0.1],         # fgsm, pgd only
        nb_iters=[10],          # pgd only
        eps_iters=[0.01],       # pgd only
        rand_inits=[False],     # pgd only
    ):
    """
    Run FGSM and PGD experiments on input model.

    Iterates over all combinations of the following parameters:
    - norms
    - epsilons          # fgsm, pgd only
    - nb_iters          # pgd only
    - eps_iters         # pgd only
    - rand_inits        # pgd only
    """

    # load existing results
    results = {}
    results_path = "attack_results/{}_{}_{}_results".format(
        model_name, dataset, attack_type
    )
    print(results_path)
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    # Eval mode
    model.eval()
    model.to(device)

    # Generate attack args
    args = process_args(mode="attack")
    args.attack_method = attack_type
    args.dataset = dataset
    args = post_process_args(args, mode="attack")

    # Loop defining the attack parameters
    for norm, epsilon, nb_iter, eps_iter, rand_init in itertools.product(
            norms, epsilons, nb_iters, eps_iters, rand_inits):
        if eps_iter > epsilon and attack_type == "pgd":
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
        print(attack_id)

        # Check if attack has already been run
        if attack_id in results:
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
        model,
        model_name,
        dataset,
        batch_size,
        epsilons,
        initial_consts=[1e-8],   # cw only
        verbose=True
    ):
    """
    Run CW attack. Similar to run but with different attack processing.
    """
    attack_type = "cw"
    norms = [2.0]

    # load existing results
    results = {}
    results_path = "attack_results/{}_{}_{}_results".format(
        model_name, dataset, attack_type
    )
    print(results_path)
    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    # Eval mode
    model.eval()
    model.to(device)

    # Generate attack args
    args = process_args(mode="attack")
    args.attack_method = attack_type
    args.dataset = dataset

    args = post_process_args(args, mode="attack")
    
    # generate model identifiers
    general_attack_id = "advens_{}_{}".format(
        attack_type, dataset
    )
    # Inner loop defining the attack parameters
    for norm, initial_const in itertools.product(norms, initial_consts):

        args.norm = norm
        args.initial_const = initial_const

        # Generate identifier for this attack
        attack_id = general_attack_id + "_{}_{}_{}".format(
            norm, initial_const, epsilons[0]
        )
        # Check if attack has already been run
        if attack_id in results:
            continue
        
        # Run attack
        test_acc, _ = evaluate_cw_l2(args, model, epsilons=epsilons)

        for epsilon, lacc in zip(epsilons, test_acc):
            attack_id = general_attack_id + "_{}_{}_{}".format(
                norm, initial_const, epsilon
            )
            # Save results
            results[attack_id] = (
                attack_type, dataset, epsilon, initial_const,
                lacc.detach().cpu().item(), norm
            )

        # Store results
        write_results(results_path, results, dictionary=True, overwrite=True)

def experiment_fgsm():
    norm = np.inf
    epsilons = [x/256 for x in [2, 5, 10, 16]]
    advens_model = load_advens_model()
    fastadv_model = load_fastadv_model()

    for model, model_name in [(advens_model, "advens"), (fastadv_model, "fastadv")]:
        # run FGSM attack
        attack_results = run(
            model=model,
            model_name=model_name,
            attack_type="fgsm",
            dataset="cifar10",
            batch_size=64,
            norms=[norm],
            epsilons=epsilons,
        )

def experiment_pgd():
    # run PGD attack (Linf)
    nb_iters = [40]
    rand_inits = [True]
    norm = np.inf
    epsilons = [x/256 for x in [2, 5, 10, 16]]
    eps_iters = [5e-4]
    
    advens_model = load_advens_model()
    fastadv_model = load_fastadv_model()

    for model, model_name in [(advens_model, "advens"), (fastadv_model, "fastadv")]:
        # run PGD attack
        attack_results = run(
            model=model,
            model_name=model_name,
            attack_type="pgd",
            dataset="cifar10",
            batch_size=64,
            norms=[norm],
            epsilons=epsilons,
            eps_iters=eps_iters,
            nb_iters=nb_iters,
            rand_inits=rand_inits,
        )

def experiment_cw():
    # run CW attack (L2)
    norm = 2.0
    epsilons = [0.5, 1.0, 2.0, 3.5]

    advens_model = load_advens_model()
    fastadv_model = load_fastadv_model()

    for model, model_name in [(advens_model, "advens"), (fastadv_model, "fastadv")]:
        # run CW attack
        attack_results = run_cw(
            model=model,
            model_name=model_name,
            dataset="cifar10",
            batch_size=64,
            epsilons=epsilons,
            initial_consts=[1e-8]
        )

if __name__ == "__main__":
    torch.manual_seed(0)

    experiment_fgsm()
    experiment_pgd()
    experiment_cw()


