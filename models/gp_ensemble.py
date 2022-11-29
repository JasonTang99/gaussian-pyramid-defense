import torch
import torch.nn as nn
from torchvision import transforms, models
import os

from utils import create_resnet, calc_resize_shape, read_results, write_results

class GPEnsemble(nn.Module):
    """
    An implementation of the proposed Gaussian Pyramid Ensemble model

    args:
        input_size (int): Size of input images.
        num_classes (int): Number of classes in dataset.

        model_folder (str): Folder to save model to.
        model_paths (list): List of model paths to save to.
        archs (list): List of model architectures used.
        device (str): Device to use for training. One of ['cpu', 'cuda'].

        up_samplers (int): Number of up-sampling ensemble models.
        down_samplers (int): Number of down-sampling ensemble models.
        interpolation (int): Interpolation method to use for resizing.
        scaling_factor (int): Scaling factor to use for resizing.
        voting_method (str): Voting method to use for ensemble.
            One of ['simple_avg', 'weighted_avg', 'majority_vote', 'weighted_vote'].
    """
    def __init__(self, args):
        super(GPEnsemble, self).__init__()
        
        self.up_samplers = args.up_samplers
        self.down_samplers = args.down_samplers
        self.interpolation = args.interpolation
        self.scaling_factor = args.scaling_factor
        self.voting_method = args.voting_method

        self.input_size = args.input_size
        self.num_classes = args.num_classes

        self.device = args.device
        self.archs = args.archs
        self.model_paths = args.model_paths
        self.model_folder = args.model_folder
        
        # Initialize models
        self.models = []
        for arch, model_path in zip(self.archs, self.model_paths):
            model = create_resnet(arch, self.num_classes, self.device)
            model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            model.eval()
            self.models.append(model)
        
        # Initialize transformations
        self.transforms = []
        for i in range(-self.down_samplers, self.up_samplers + 1):
            transform = transforms.Compose([
                transforms.Resize(
                    calc_resize_shape(
                        in_size=self.input_size,
                        scaling_exp=i,
                        scaling_factor=self.scaling_factor
                    ),
                    interpolation=self.interpolation,
                    antialias=True
                ),
            ])
            self.transforms.append(transform)
        
        # Calculate model weights from validation accuracies
        if "weighted" in self.voting_method:
            val_acc_map = read_results(self.model_folder)
            val_accs = torch.tensor(
                [val_acc_map[model_path] for model_path in self.model_paths]
            )
            self.model_weights = val_accs / val_accs.sum()

    def forward(self, x):
        # apply transformations and forward pass through models
        outputs = [
            model(t(x)) for model, t in zip(self.models, self.transforms)
        ]
        
        # average outputs
        if self.voting_method == "simple_avg":
            output = torch.mean(torch.stack(outputs), dim=0)
        elif self.voting_method == "weighted_avg":
            output = torch.sum(
                torch.stack(outputs) * self.model_weights.view(-1, 1, 1),
                dim=0
            )

            # TODO: check this
            
        # elif self.voting_method == "majority_vote":
        #     output = torch.mean(torch.stack(outputs), dim=0)


        

        return output
