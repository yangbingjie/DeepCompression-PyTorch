import torch
import os
import math
from pruning.function.Prune import MaskConv2Module
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
from scipy.sparse import csr_matrix

# def filer_zero(value, index, bits):
#     max_bits = 2 ** bits
#     last_index = -1
#     i = 0
#     while i < len(index):
#         diff = index[i] - last_index
#         if diff > max_bits:
#             filer_num = int(diff / max_bits)
#             for j in range(filer_num):
#                 value = np.insert(value, i, 0)
#                 index = np.insert(index, i, max_bits - 1)
#             last_index += filer_num * max_bits
#             i += filer_num
#         else:
#             last_index = index[i]
#             index[i] = diff - 1
#             i += 1
#     return value, index
#
#
# # test csr
# a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.4, 0, 0,
#               0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.7])
# tensor = torch.from_numpy(a)
# mat = csr_matrix(tensor)
# print(mat.data)
# print(mat.indices)
# data, indices = filer_zero(mat.data, mat.indices, 3)
# print(data)
# print(indices)


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parallel_gpu = False
use_cuda = torch.cuda.is_available()
train_batch_size = 128
train_epoch = 256
retrain_epoch = 8
retrain_num = 16
test_batch_size = 128
accuracy_accept = 92
lr = 0.01
momentum = 0.9
# valid_size = 0.001
learning_rate_decay = 0.0005
train_path_root = './pruning/result/'
train_path_name = 'VGG16'
train_path = train_path_root + train_path_name
retrain_path = './pruning/result/VGG16_retrain'
data_dir = './data'

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=train_transform)

# validset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
#                                         download=True, transform=train_transform)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=test_transform)

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

# print('length of trainset, validset and testset')
# print(len(trainloader))
# print(len(valid_loader))
# print(len(testloader))


net = VGG16(num_classes=10)
# net = torchvision.models.vgg16(pretrained=False)
optimizer = optim.SGD(list(net.cuda().parameters()), lr=lr, momentum=momentum, weight_decay=learning_rate_decay)
criterion = nn.CrossEntropyLoss()
if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    if parallel_gpu:
        # parallel use GPU
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
    # speed up slightly
    cudnn.benchmark = True

if not os.path.exists(train_path_root):
    os.mkdir(train_path_root)

# if os.path.exists(train_path):
#     net.load_state_dict(torch.load(train_path))
# else:
helper.train(testloader, net, trainloader=trainloader, criterion=criterion,
             optimizer=optimizer, epoch=train_epoch, accuracy_accept=accuracy_accept, train_path=train_path)

# helper.test(testloader, net)
# log.log_file_size(train_path, 'M')

# for j in range(retrain_num):
#     retrain_mode = 'conv' if j % 2 == 1 else 'fc'
#     # We used five iterations of pruning an retraining
#     for k in range(5):
#         if parallel_gpu:
#             net.module.prune_layer(prune_mode=retrain_mode)
#         else:
#             net.prune_layer(prune_mode=retrain_mode)
#     print('====================== Retrain', retrain_mode, j, 'Start ==================')
#     if retrain_mode == 'fc':
#         if parallel_gpu:
#             net.module.compute_dropout_rate()
#         else:
#             net.compute_dropout_rate()
#     if parallel_gpu:
#         net.module.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
#     else:
#         net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
#     x = filter(lambda p: p.requires_grad, list(net.parameters()))
#     optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(net.parameters())), lr=lr / 100, momentum=momentum,
#                           weight_decay=learning_rate_decay)
#     helper.train(testloader, net, trainloader=trainloader, criterion=criterion,, auto_sava=False,
#                  optimizer=optimizer, epoch=retrain_epoch, accuracy_accept=accuracy_accept,train_path=retrain_path)
#     # helper.test(testloader, net)
#     print('====================== ReTrain End ======================')
#
#     helper.save_sparse_model(net, retrain_path)
#     log.log_file_size(retrain_path, 'M')
