import torch
import os
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.AlexNet import AlexNet
import pruning.function.helper as helper
import torchvision.datasets as datasets
import util.log as log
import torch.optim as optim

use_cuda = True
seed = 46
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(seed)
else:
    print('Not using CUDA!!!')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
batch_size = 512

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

criterion = nn.CrossEntropyLoss()

retrain_num = 2
train_path = './pruning/result/AlexNet'
retrain_path = './pruning/result/AlexNet_retrain'

net = AlexNet(num_classes=10).to(device)
if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path))
lr = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
log.log_file_size(train_path, 'M')
helper.test(testloader, net)
net.load_state_dict(torch.load(train_path))
net.eval()
for j in range(retrain_num):
    retrain_mode = 'conv' if j % 2 == 0 else 'fc'
    net.prune_layer(prune_mode=retrain_mode)
    print('====================== Retrain', retrain_mode, j, 'Start ==================')
    net.compute_dropout_rate()
    # net.fix_layer(fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr / 100, momentum=0.9,
                                weight_decay=1e-5)
    helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
    torch.save(net.state_dict(), retrain_path)
    log.log_file_size(retrain_path, 'M')
    helper.test(testloader, net)
    print('====================== ReTrain End ======================')
