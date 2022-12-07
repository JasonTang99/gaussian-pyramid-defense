import os
import numpy as np
import itertools
import torch

from parse_args import process_args, post_process_args
from attack import evaluate_attack, evaluate_cw_l2
from models.gp_ensemble import GPEnsemble
from models.denoisers import DnCNN
from utils import read_results, write_results

from tqdm import tqdm

def run(attack_type,
        dataset,
        scaling_factor,
        batch_size,
        up_down_pairs,
        interpolations,
        voting_methods,
        norms=[np.inf],
        epsilons=[0.1],
        nb_iters=[10],
        eps_iters=[0.01],
        rand_inits=[False],
        initial_consts=[0.1],
        verbose=True,
        use_denoiser=True):
    """
    Run FGSM and PGD experiments on GPEnsemble models.

    Tests 2 models (linear and voting) since we use the linear model
    to generate the adversarial examples for the non-differentiable
    voting model.

    Iterates over all combinations of the following parameters:
    - up_samplers, down_samplers from up_down_pairs
    - interpolations
    - voting_methods
    - norms
    - epsilons          # fgsm, pgd only
    - nb_iters          # pgd only
    - eps_iters         # pgd only
    - rand_inits        # pgd only
    - initial_consts    # cw only
    """
    # load existing results
    results = {}
    if use_denoiser:
        results_path = "attack_results/{}_{}_{}_denoiser_results".format(
            dataset, attack_type, scaling_factor
        )
    else:
        results_path = "attack_results/{}_{}_{}_results".format(
            dataset, attack_type, scaling_factor
        )

    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    # Setup tqdm
    total_experiments = len(up_down_pairs) * len(interpolations) * len(voting_methods) \
        * len(norms) * len(epsilons) * len(nb_iters) * len(eps_iters) \
        * len(rand_inits) * len(initial_consts)
    pbar = tqdm(total=total_experiments)
    
    # outer loop defining the model
    for (up_samplers, down_samplers), interpolation, voting_method in itertools.product(
            up_down_pairs, interpolations, voting_methods):
        
        # generate model identifiers
        diffable_voting = 'simple_avg' if voting_method == 'simple' else 'weighted_avg'
        nondiffable_voting = 'majority_vote' if voting_method == 'simple' else 'weighted_vote'
        
        linear_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            diffable_voting
        )
        voting_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            nondiffable_voting
        )
            
        # Generate attack args
        linear_args = process_args(mode="attack")
        voting_args = process_args(mode="attack")

        # Decrease batch size if using 2x up-sampling
        if up_samplers >= 2 and scaling_factor == 2:
            batch_size = 16
            print("batch_size: ", batch_size)
        
        for args in [linear_args, voting_args]:
            args.attack_method = attack_type
            args.dataset = dataset
            args.scaling_factor = scaling_factor

            args.up_samplers = up_samplers
            args.down_samplers = down_samplers
            args.interpolation = interpolation
            args.archs = ['resnet18']
            args.batch_size = batch_size
        
        linear_args.voting_method = diffable_voting
        voting_args.voting_method = nondiffable_voting

        # Post process args
        linear_args = post_process_args(linear_args, mode="attack")
        voting_args = post_process_args(voting_args, mode="attack")

        # Generate models
        linear_model, voting_model = None, None

        # Inner loop defining the attack parameters
        for norm, epsilon, nb_iter, eps_iter, rand_init, initial_const in itertools.product(
                norms, epsilons, nb_iters, eps_iters, rand_inits, initial_consts):
            pbar.update(1)
            if eps_iter > epsilon and attack_type == "pgd":
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

            # Load DnCNN denoiser
            if use_denoiser:
                denoiser = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(linear_args.device)
                dn_path = os.path.join('trained_denoisers', f'dncnn_{dataset}_mixed+gaussian.pth')
                denoiser.load_state_dict(torch.load(dn_path, map_location=linear_args.device))
            else:
                denoiser = None

            # Run attack
            linear_acc, voting_acc = evaluate_attack(linear_args, linear_model, voting_model, denoiser)

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
                diffable_voting, linear_acc.detach().cpu().item(), norm
            )
            results[voting_attack_id] = (
                attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                interpolation, epsilon, nb_iter, eps_iter, rand_init, initial_const,
                nondiffable_voting, voting_acc.detach().cpu().item(), norm
            )

            # Store results
            write_results(results_path, results, dictionary=True, overwrite=True)
    pbar.close()

def run_cw(
        dataset,
        scaling_factor,
        batch_size,
        up_down_pairs,
        voting_methods,
        epsilons,
        initial_consts=[1e-8],   # cw only
        verbose=True,
        use_denoiser=True
    ):
    """
    Run CW attack. Similar to run but with different attack processing.
    """
    attack_type = "cw"
    norms = [2.0]
    interpolations = ['bilinear']

    # load existing results
    results = {}
    if use_denoiser:
        results_path = "attack_results/{}_{}_{}_denoiser_results".format(
            dataset, attack_type, scaling_factor
        )
    else:
        results_path = "attack_results/{}_{}_{}_results".format(
            dataset, attack_type, scaling_factor
        )

    if os.path.exists(results_path):
        results = read_results(results_path)
        print(f"Loaded {len(results)} results")
    
    total_experiments = len(up_down_pairs) * len(voting_methods) * len(initial_consts)
    pbar = tqdm(total=total_experiments)
    # outer loop defining the model
    for (up_samplers, down_samplers), interpolation, voting_method in itertools.product(
            up_down_pairs, interpolations, voting_methods):
        
        # generate model identifiers
        diffable_voting = 'simple_avg' if voting_method == 'simple' else 'weighted_avg'
        nondiffable_voting = 'majority_vote' if voting_method == 'simple' else 'weighted_vote'
        
        linear_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            diffable_voting
        )
        voting_model_id = "{}_{}_{}_{}_{}_{}_{}".format(
            attack_type,
            dataset,
            scaling_factor,
            up_samplers,
            down_samplers,
            interpolation,
            nondiffable_voting
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
            args.batch_size = batch_size
        
        linear_args.voting_method = diffable_voting
        voting_args.voting_method = nondiffable_voting

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
                norm, epsilons[0], initial_const
            )
            # Check if attack has already been run
            if attack_id in results:
                print("Attack already run, skipping...")
                continue
            
            # Load models if not already loaded
            if linear_model is None:
                linear_model = GPEnsemble(linear_args)
                voting_model = GPEnsemble(voting_args)

            # Load DnCNN denoiser
            if use_denoiser:
                denoiser = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(linear_args.device)
                dn_path = os.path.join('trained_denoisers', f'dncnn_{dataset}_mixed+gaussian.pth')
                denoiser.load_state_dict(torch.load(dn_path, map_location=linear_args.device))
            else:
                denoiser = None

            # Run attack
            linear_acc, voting_acc = evaluate_cw_l2(linear_args, 
                linear_model, voting_model, denoiser, epsilons=epsilons)

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
                    diffable_voting, lacc.detach().cpu().item(), norm
                )
                results[voting_attack_id] = (
                    attack_type, dataset, scaling_factor, up_samplers, down_samplers,
                    interpolation, epsilon, initial_const,
                    nondiffable_voting, vacc.detach().cpu().item(), norm
                )

            # Store results
            write_results(results_path, results, dictionary=True, overwrite=True)
    pbar.close()

def experiment_fgsm():
    norm = np.inf
    epsilons = [x/256 for x in [2, 5, 10, 16]]
    interpolations = ["bilinear"]
    voting_methods = ['simple', 'weighted']

    scalings, datasets, denoisers = [1.1, 2.0], ["mnist", "cifar10"], [True, False]

    for scaling, dataset, denoiser in itertools.product(scalings, datasets, denoisers):
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
        if denoiser:
            print("Using DnCNN denoiser")
        
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
            verbose=True,
            denoiser=denoiser
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
    
    scalings, datasets, denoisers = [1.1, 2.0], ["mnist", "cifar10"], [True, False]

    for scaling, dataset, denoiser in itertools.product(scalings, datasets, denoisers):
        if scaling == 2.0:
            up_down_pairs = [(i, i) for i in [0, 3]]
        else:
            up_down_pairs = [(i, i) for i in [3, 5, 7]]

        print("Running PGD attack on {} {} {}".format(dataset, scaling, norm))
        if denoiser:
            print("Using DnCNN denoiser")

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
            verbose=False,
            denoiser=denoiser,
        )

def experiment_cw():
    # run CW attack (L2)
    norm = 2.0
    epsilons = [0.5, 1.0, 2.0, 3.5]
    interpolations = ["bilinear"]
    voting_methods = ['simple', 'weighted']
    
    scalings, datasets, denoisers = [1.1, 2.0], ["mnist", "cifar10"], [True, False]

    for scaling, dataset, denoiser in itertools.product(scalings, datasets, denoisers):    
        if scaling == 2.0:
            up_down_pairs = [(i, i) for i in [0, 3]]
        else:
            up_down_pairs = [(i, i) for i in [5, 7]]

        print("Running CW attack on {} {} {}".format(dataset, scaling, norm))
        if denoiser:
            print("Using DnCNN denoiser")
            
        attack_results = run_cw(
            dataset=dataset,
            scaling_factor=scaling,
            batch_size=32,
            up_down_pairs=up_down_pairs,
            voting_methods=voting_methods,
            epsilons=epsilons,
            verbose=False,
            denoiser=denoiser
        )

if __name__ == "__main__":
    experiment_fgsm()
    experiment_pgd()
    experiment_cw()


