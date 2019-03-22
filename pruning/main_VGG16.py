import torch
import os
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from pruning.net.VGG16 import VGG16
import pruning.function.helper as helper
import util.log as log
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

use_cuda = torch.cuda.is_available()
train_batch_size = 1024
retrain_num = 8
train_epoch = 64
retrain_epoch = 8
test_batch_size = 64
loss_accept = 1e-2
lr = 1e-2
valid_size = 0.1
data_dir = './data'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)

validset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)


num_test = len(testset)
indices = list(range(num_test))
split = int(np.floor(valid_size * num_test))

test_idx, valid_idx = indices[split:], indices[:split]
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True,
                                          **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=test_batch_size,
                                           sampler=valid_sampler,
                                           **kwargs)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         sampler=test_sampler, **kwargs)

print('length of trainset, validset and testset')
print(len(trainloader))
print(len(valid_loader))
print(len(testloader))

train_path = './pruning/result/VGG16'
retrain_path = './pruning/result/VGG16_retrain'

net = VGG16(num_classes=10)
optimizer = optim.SGD(list(net.cuda().parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# net = torchvision.models.vgg16(pretrained=False)

if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    # parallel use GPU
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
    # speed up slightly
    cudnn.benchmark = True

if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path))
else:
    helper.train(net, trainloader=trainloader, valid_loader=valid_loader, criterion=criterion,
                 optimizer=optimizer, epoch=train_epoch, loss_accept=loss_accept)
    torch.save(net.state_dict(), train_path)

helper.test(testloader, net)
log.log_file_size(train_path, 'M')

for j in range(retrain_num):
    retrain_mode = 'conv' if j % 2 == 1 else 'fc'
    # We used five iterations of pruning an retraining
    for k in range(5):
        if use_cuda:
            net.module.prune_layer(prune_mode=retrain_mode)
        else:
            net.prune_layer(prune_mode=retrain_mode)
    print('====================== Retrain', retrain_mode, j, 'Start ==================')
    if retrain_mode == 'fc':
        if use_cuda:
            net.module.compute_dropout_rate()
        else:
            net.compute_dropout_rate()
    if use_cuda:
        net.module.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    else:
        net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    x = filter(lambda p: p.requires_grad, list(net.parameters()))
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(net.parameters())), lr=lr / 100, momentum=0.9,
                          weight_decay=1e-5)
    helper.train(net, trainloader=trainloader, valid_loader=valid_loader, criterion=criterion,
                 optimizer=optimizer, epoch=retrain_epoch, loss_accept=loss_accept)
    helper.test(testloader, net)
    helper.save_sparse_model(net, retrain_path)
    log.log_file_size(retrain_path, 'M')
    print('====================== ReTrain End ======================')
