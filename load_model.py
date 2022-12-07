import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from comparison_defenses.ens_adv_train.models.cifar10.resnet import ResNet18
from comparison_defenses.fast_adversarial.preact_resnet import PreActResNet18
from models.resnet import MyResNet18

def load_advens_model(fp="trained_models/comparison/resnet18_ens_adv.pth"):
    model = ResNet18()
    model.load_state_dict(torch.load(fp))
    return model

def load_fastadv_model(fp="trained_models/comparison/cifar_model_weights_30_epochs.pth"):
    model = PreActResNet18()
    model.load_state_dict(torch.load(fp))
    return model

def load_resnet(arch="resnet18", num_classes=10, device="cpu", pretrained=True, grayscale=False):
    if arch == "resnet18":
        model = MyResNet18(grayscale=grayscale)
    else:
        raise ValueError("Model not supported.")
    
    # load pretrained weights
    if pretrained:
        model.load_state_dict(resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        ).state_dict())

    # change output size
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # move to device
    model = model.to(device)

    return model

if __name__ == "__main__":
    model = load_advens_model()
    print(model)
    model = load_fastadv_model()
    print(model)
    model = load_resnet()
    print(model)