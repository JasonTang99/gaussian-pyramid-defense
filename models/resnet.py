import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms

class MyResNet18(ResNet):
    def __init__(self, grayscale=False):
        super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.grayscale = grayscale
        self.grayscale_transform = transforms.Grayscale(num_output_channels=3)
        
    def forward(self, x):
        # Grayscale before transform
        if self.grayscale:
            x = self.grayscale_transform(x)
        return super(MyResNet18, self).forward(x)
