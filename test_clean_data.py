import torch
import numpy as np
from utils import *
from models.gp_ensemble import GPEnsemble
from parse_args import *
from load_model import load_advens_model, load_resnet, load_fastadv_model
from datasets import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on(model, dataset='mnist', in_size=None, normalize=False):
    """
    Evaluate input model on clean data.
    """
    model.to(device)
    # setup args
    args = process_args("attack")
    args.dataset = dataset
    args = post_process_args(args, "attack")

    # load dataset
    test_loader = load_data(args, train=False, normalize=normalize)

    # evaluate model
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Accuracy: {}".format(correct / len(test_loader.dataset)))

if __name__ == "__main__":
    # FastAdversarial test
    model = load_fastadv_model()
    evaluate_on(model, dataset='cifar10', normalize=True)

    # Advens test
    model = load_advens_model()
    evaluate_on(model, dataset='cifar10')

    # Single resnet test
    model = load_resnet(arch="resnet18", pretrained=True, num_classes=10, grayscale=False)
    model.load_state_dict(torch.load("trained_models/cifar10/resnet18_1.1+0_BL.pth"))
    evaluate_on(model, dataset='cifar10')

    # GPEnsemble test
    for vm in "simple_avg weighted_avg majority_vote weighted_vote".split():
        args = process_args("attack")
        args.dataset = "cifar10"
        args.up_samplers = 2
        args.down_samplers = 2
        args.voting_method = vm
        args = post_process_args(args, "attack")
        model = GPEnsemble(args)

        print(vm, end=" ")

        evaluate_on(model, args.dataset)
