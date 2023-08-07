import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import os

from load_model import load_resnet
from utils import calc_resize_shape, read_results, write_results

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
        self.antialias = args.antialias
        self.scaling_factor = args.scaling_factor
        self.voting_method = args.voting_method

        self.input_size = args.input_size
        self.num_classes = args.num_classes

        self.device = args.device
        self.archs = args.archs
        self.model_paths = args.model_paths
        self.model_folder = args.model_folder

        self.grayscale = args.dataset == "mnist"
        
        # Initialize models
        self.models = []
        for arch, model_path in zip(self.archs, self.model_paths):
            model = load_resnet(arch, self.num_classes, self.device, 
                grayscale=self.grayscale)
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
                    antialias=self.antialias
                ),
            ])
            self.transforms.append(transform)
        
        # Calculate model weights from validation accuracies
        if "weighted" in self.voting_method:
            val_acc_map = read_results(os.path.join(self.model_folder, "results"))
            val_accs = torch.tensor(
                [val_acc_map[model_path] for model_path in self.model_paths],
                device=self.device
            )
            self.model_weights = val_accs / val_accs.sum()

    def forward(self, x):
        # apply transformations and forward pass through models
        # (num_models, batch_size, num_classes)
        outputs = torch.stack([
            model(t(x)) for model, t in zip(self.models, self.transforms)
        ])

        # average outputs
        if self.voting_method == "simple_avg":
            output = torch.mean(outputs, dim=0)
        elif self.voting_method == "weighted_avg":
            output = torch.sum(
                outputs * self.model_weights.view(-1, 1, 1),
                dim=0
            )
        elif self.voting_method == "majority_vote":
            # Get votes (num_models, batch_size)
            outputs = torch.argmax(outputs, dim=2)

            # Select mode and one-hot encode (batch_size, num_classes)
            output = F.one_hot(
                torch.mode(outputs, dim=0).values,
                num_classes=self.num_classes
            )
        elif self.voting_method == "weighted_vote":
            # Get votes (num_models, batch_size)
            outputs = torch.argmax(outputs, dim=2)

            # One-hot encode (num_models, batch_size, num_classes)
            outputs = F.one_hot(outputs, num_classes=self.num_classes)

            # Weighted sum (batch_size, num_classes)
            outputs = torch.sum(
                outputs * self.model_weights.view(-1, 1, 1),
                dim=0
            )
            # Set highest probability to 1 and rest to 0
            output = torch.where(
                outputs == torch.max(outputs, dim=1, keepdim=True).values,
                1.0, 0.0
            )
        else:
            raise ValueError("Invalid voting method", self.voting_method)

        return output
