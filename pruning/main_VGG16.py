import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from pruning.net.VGG16 import VGG16
import pruning.function.helper as helper
import util.log as log
import torch.optim as optim

use_cuda = torch.cuda.is_available()
batch_size = 256
retrain_num = 2
train_epoch = 8
retrain_epoch = 2
lr = 1e-3
train_log_frequency = 50
retrain_log_frequency = 50
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, **kwargs)

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
    helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epoch=train_epoch,
                 log_frequency=train_log_frequency)
    torch.save(net.state_dict(), train_path)

log.log_file_size(train_path, 'M')
# helper.test(testloader, net)


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
    helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epoch=retrain_epoch,
                 log_frequency=retrain_log_frequency)
    helper.test(testloader, net)
    helper.save_sparse_model(net, retrain_path)
    log.log_file_size(retrain_path, 'M')
    print('====================== ReTrain End ======================')
