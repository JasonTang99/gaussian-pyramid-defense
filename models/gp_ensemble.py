import torch
import torch.nn as nn

class GPEnsemble(nn.Module):
    """
    An implementation of the proposed Gaussian Pyramid Ensemble model
    """
    def __init__(self, args, models):
        super(GPEnsemble, self).__init__()
        
        # setup up-sampling models
        self.up_samplers = []

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same'))
        for i in range(hidden_layers):
            layers.append(ResNetBlock(hidden_channels, hidden_channels, kernel_size=kernel_size))

        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
