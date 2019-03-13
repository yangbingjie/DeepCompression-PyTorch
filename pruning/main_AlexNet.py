import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.AlexNet import AlexNet
from pruning.function.helper import train, test

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='../data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

criterion = nn.CrossEntropyLoss()

retrain_num = 3
path = './result/LeNet'
net = AlexNet()
train(net, trainloader=trainloader, criterion=criterion, is_retrain=False, path=path)
test(testloader, net)

for j in range(retrain_num):
    train(net, trainloader=trainloader, criterion=criterion, retrain_num=j + 1, path=path)
    test(testloader, net)



