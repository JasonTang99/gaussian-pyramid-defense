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


# def create_resnet(arch="resnet18", num_classes=10, device="cpu", pretrained=True, grayscale=False):
#     if arch == "resnet18":
#         model = MyResNet18(grayscale=grayscale)
#     else:
#         raise ValueError("Model not supported.")
    
#     # load pretrained weights
#     if pretrained:
#         model.load_state_dict(resnet18(
#             weights=ResNet18_Weights.IMAGENET1K_V1
#         ).state_dict())

#     # change output size
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

#     # move to device
#     model = model.to(device)

#     return model




# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))

#     return model

# weights = ResNet18_Weights.verify(weights)
# _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


# Generate resnet model
# def create_resnet(arch="resnet18", num_classes=10, device="cpu", pretrained=True):
#     """
#     Create resnet model.
#     """
#     if arch == "resnet18":
#         model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 
#                          if pretrained else None)
#     elif arch == "resnet34":
#         model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1
#                          if pretrained else None)
#     elif arch == "resnet50":
#         model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1
#                          if pretrained else None)
#     else:
#         raise ValueError("Model not supported.")
    
#     # change output size
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

#     # freeze all non-fc layers
#     # for param in model.parameters():
#     #     param.requires_grad = False
#     # for param in model.fc.parameters():
#     #     param.requires_grad = True
    
#     # move model to device
#     model = model.to(device)
    
#     return model