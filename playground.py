import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import transforms, models
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import InterpolationMode

from models.utils import create_resnet

model = create_resnet(device="cuda", output_size=10, model="resnet18")
model.load_state_dict(torch.load("trained_models/mnist/resnet18_-1_BL.pth"))
model.eval()
model = model.cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])
train_data = MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# evaluate model
t = transforms.Compose([
    transforms.Resize(14, interpolation=InterpolationMode.BILINEAR, antialias=True),
])

correct = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data = t(data)

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print("Accuracy: {}".format(correct / len(train_data)))
exit(0)

# train_data = MNIST(root='data', train=True, download=True)

# t1 = transforms.Compose([
#     transforms.Resize(14, interpolation=InterpolationMode.BILINEAR),
#     transforms.ToTensor(),
# ])
# t2 = transforms.Compose([
#     transforms.ToTensor(),
# ])
# i1 = t1(train_data[0][0])
# i2 = t2(train_data[0][0])
# print(i1.shape)
# print(i2.shape)

# # half the size of the image 2
# i2 = F.interpolate(i2.unsqueeze(0), scale_factor=0.5, mode='bilinear')
# print(i2.shape)

# # check all close
# print(torch.allclose(i1, i2.squeeze(0)))

# i_diff = i1 - i2.squeeze(0)
# print(torch.max(i_diff))
# print(torch.max(i2))

# # plot all
# import matplotlib.pyplot as plt
# plt.subplot(1, 3, 1)
# plt.imshow(i1.permute(1, 2, 0))
# plt.subplot(1, 3, 2)
# plt.imshow(i2.squeeze(0).permute(1, 2, 0))
# plt.subplot(1, 3, 3)
# plt.imshow(i_diff.permute(1, 2, 0))
# plt.show()



# exit(0)




# create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# evaluate model
correct = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print("Accuracy: {}".format(correct / len(val_data)))

# correct = 0
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.cuda(), target.cuda()
#         output = model(data)
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
# print("Train Accuracy: {}".format(correct / len(train_data)))




transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                            shuffle=True, num_workers=2)    

transform2 = transforms.Compose(
    [        
        transforms.Resize(14, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])

# evaluate model


correct = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        # print(images.shape)
        images = transform2(images)

        images = images.cuda()
        labels = labels.cuda()

 
        # images = F.interpolate(images, size=(14, 14), mode='bilinear')

        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f'Accuracy of the network on the train images: {100 * correct / len(trainset)}')

# # half the image size

# print(images.shape)

# # forward pass
# y = model(images)
# print(y.shape)

# # calculate accuracy
# _, predicted = torch.max(y.data, 1)
# correct = (predicted == labels).sum().item()
# print(correct)
