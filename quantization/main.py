import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from pruning.net.PruneLeNet5 import PruneLeNet5
from quantization.function.weight_share import share_weight
from quantization.net.LeNet5 import LeNet5
import quantization.function.helper as helper
import torch.optim as optim
from pruning.function.helper import test
import torch.backends.cudnn as cudnn
import util.log as log

prune_result_path = './pruning/result/LeNet_retrain'

retrain_codebook_root = './quantization/result/'
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)
retrain_codebook_name = 'LeNet_codebook'
retrain_epoch = 1

use_cuda = torch.cuda.is_available()
train_batch_size = 1
test_batch_size = 32
parallel_gpu = False
lr = 1e-4
prune_fc_bits = 4
quantization_conv_bits = 8
quantization_fc_bits = 4
data_dir = './data'
retrain_codebook_path = retrain_codebook_root + retrain_codebook_name
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)

prune_net = PruneLeNet5()
conv_layer_length, codebook, nz_num, conv_diff, fc_diff = share_weight(
    prune_net, prune_result_path, quantization_conv_bits, quantization_fc_bits, prune_fc_bits)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
# Loader
kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                      download=True, transform=transform)

testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                     download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True,
                                          **kwargs)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         **kwargs)

net = LeNet5()
max_value = np.finfo(np.float32).max
max_conv_bit = 2 ** quantization_conv_bits
max_fc_bit = 2 ** quantization_fc_bits
index_list, key_parameter = helper.sparse_to_init(
    net, conv_layer_length, nz_num, conv_diff, fc_diff, codebook, max_conv_bit, max_fc_bit)
if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    if parallel_gpu:
        # parallel use GPU
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
    # speed up slightly
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

test(use_cuda, testloader, net)

helper.train_codebook(key_parameter, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length, codebook,
                      index_list, testloader, net, trainloader, criterion,
                      optimizer, retrain_epoch)

helper.save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, retrain_codebook_path, net)

log.log_file_size(retrain_codebook_path, 'K')
