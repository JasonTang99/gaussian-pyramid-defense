import torch
import torchvision

# # load resnet 18
# resnet18 = torchvision.models.resnet18(pretrained=True)

# # create sample input
# x = torch.randn(32, 3, 224, 224)

# # forward pass
# y = resnet18(x)
# print(resnet18)
# print(y.shape)

# newmodel = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
# print(newmodel)
# y = newmodel(x)
# print(y.shape)

# x = torch.randn(32, 3, 32, 32)
# y = resnet18(x)
# print(y.shape)

from models.utils import create_resnet

model = create_resnet(device="cuda", output_size=10, model="resnet18")
# load weights from trained model
model.load_state_dict(torch.load("trained_models/mnist/up_1_resnet18.pth"))
model.eval()

# model to cuda
model = model.cuda()

# create sample input
x = torch.randn(32, 3, 28, 28).cuda()

# forward pass
y = model(x)
print(y.shape)

