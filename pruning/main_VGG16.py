import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pruning.net.VGG16 import VGG16
import pruning.function.helper as helper
import torch.multiprocessing as multiprocessing
import util.log as log
import torch.optim as optim

use_cuda = torch.cuda.is_available()
seed = 42

device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Not using CUDA!!!')
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
multiprocessing.set_start_method('spawn')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32
retrain_num = 4
train_epoch = 32
retrain_epoch = 4
lr = 1e-2
log_frequency = 1000

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

criterion = nn.CrossEntropyLoss()

train_path = './pruning/result/VGG16'
retrain_path = './pruning/result/VGG16_retrain'

net = VGG16(num_classes=10)
if use_cuda:
    net.cuda()
    criterion.cuda()
optimizer = optim.SGD(list(net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
# if os.path.exists(train_path):
#     net.load_state_dict(torch.load(train_path))
helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epoch=train_epoch, log_frequency=log_frequency)
torch.save(net.state_dict(), train_path)
# log.log_file_size(train_path, 'M')
# helper.test(testloader, net)
# net.load_state_dict(torch.load(train_path))
# net.eval()
#
# for j in range(retrain_num):
#     retrain_mode = 'conv' if j % 2 == 0 else 'fc'
#     # We used five iterations of pruning an retraining
#     for k in range(5):
#         net.prune_layer(prune_mode=retrain_mode)
#     print('====================== Retrain', retrain_mode, j, 'Start ==================')
#     if retrain_mode == 'fc':
#         net.compute_dropout_rate()
#     net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
#     x = filter(lambda p: p.requires_grad, list(net.parameters()))
#     optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(net.parameters())), lr=lr/100, momentum=0.9, weight_decay=1e-5)
#     helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer, log_frequency=log_frequency)
#     helper.save_sparse_model(net, retrain_path)
#     log.log_file_size(retrain_path, 'M')
#     helper.test(testloader, net)
#     print('====================== ReTrain End ======================')
