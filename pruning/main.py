import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.PruneLeNet5 import PruneLeNet5
import pruning.function.helper as helper
import util.log as log
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.multiprocessing as multiprocessing
from torch.utils.data.sampler import SubsetRandomSampler
import multiprocessing as mp

mp.set_start_method('spawn')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# # test csr
# a = np.array([0, 3.4, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.7])
# tensor = torch.from_numpy(a)
# csr_matrix = WeightCSR(tensor, index_bits=3)
# a, b = csr_matrix.tensor_to_csr()
# print(a)
# print(b)
# print(bin(8)[2:].zfill(3))

# device_id = 1
parallel_gpu = False
use_cuda = False  # torch.cuda.is_available()
sensitivity = {
    'conv1': 0.4,
    'conv2': 0.73,
    'fc1': 0.9,
    'fc2': 0.7
}
prune_num_per_retrain = 3
print(sensitivity)
train_batch_size = 32
test_batch_size = 64
lr = 1e-2
# valid_size = 0.3
# 330 3000 32000 950
retrain_mode_list = [
    {'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8},
    {'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8},
    {'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8},
    {'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8},
    {'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8}
]
print(retrain_mode_list)
train_epoch = 4
train_path_root = './pruning/result/'
train_path_name = 'LeNet'
train_path = train_path_root + train_path_name
if not os.path.exists(train_path_root):
    os.mkdir(train_path_root)
retrain_path = './pruning/result/LeNet_retrain'
data_dir = './data'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
# Loader
kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                      download=True, transform=transform)
#
# validset = torchvision.datasets.MNIST(root=data_dir, train=True,
#                                       download=True, transform=transform)

testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                     download=True, transform=transform)

# num_test = len(testset)
# indices = list(range(num_test))
# split = int(np.floor(valid_size * num_test))

# test_idx, valid_idx = indices[split:], indices[:split]
# test_sampler = SubsetRandomSampler(test_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True,
                                          **kwargs)
# valid_loader = torch.utils.data.DataLoader(validset, batch_size=test_batch_size,
#                                            sampler=valid_sampler,
#                                            **kwargs)

# testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
#                                          sampler=test_sampler, **kwargs)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         **kwargs)

net = PruneLeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    # torch.cuda.set_device(device_id)
    if parallel_gpu:
        # parallel use GPU
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
    # speed up slightly
    cudnn.benchmark = True

# weight_decay is L2 regularization
if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path))
else:
    helper.train(testloader, net, trainloader, criterion, optimizer, train_path, epoch=train_epoch, use_cuda=use_cuda,
                 epoch_step=25)
    torch.save(net.state_dict(), train_path)
log.log_file_size(train_path, 'K')
helper.test(use_cuda, testloader, net)

for j in range(len(retrain_mode_list)):
    retrain_mode = retrain_mode_list[j]['mode']
    for k in range(retrain_mode_list[j]['prune_num']):
        net.prune_layer(prune_mode=retrain_mode, use_cuda=use_cuda, sensitivity=sensitivity)
    print('====================== Retrain', j, retrain_mode, 'Start ==================')
    if retrain_mode_list[j]['mode'] != 'full':
        net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    # After pruning, the network is retrained with 1/10 of the original network's learning rate
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr / 10, weight_decay=1e-5)
    helper.train(testloader, net, trainloader, criterion, optimizer, retrain_path, use_cuda=use_cuda,
                 epoch=retrain_mode_list[j]['retrain_epoch'], save_sparse=True)
    print('====================== ReTrain End ======================')
    log.log_file_size(retrain_path, 'K')

# prune rate:  5.80908
# The file size is 400.81 K
