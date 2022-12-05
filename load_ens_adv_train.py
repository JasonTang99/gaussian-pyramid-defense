import torch

from ens_adv_train.models.cifar10.resnet import ResNet18

def get_ens_adv_model(fp="trained_models/resnet18_ens_adv.pth"):
    model = ResNet18()
    model.load_state_dict(torch.load(fp))
    return model

if __name__ == "__main__":
    model = get_ens_adv_model()
    print(model)