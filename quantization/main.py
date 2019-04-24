import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import util.log as log
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pruning.function.helper import test
from quantization.net.LeNet5 import LeNet5
from quantization.net.AlexNet import AlexNet
from quantization.net.VGG16 import VGG16
import quantization.function.helper as helper
from pruning.net.PruneLeNet5 import PruneLeNet5
from pruning.net.PruneVGG16 import PruneVGG16
from pruning.net.PruneAlexNet import PruneAlexNet

from pruning.function.helper import load_dataset
from quantization.function.weight_share import share_weight

parser = argparse.ArgumentParser()
parser.add_argument("net", help="Network name", type=str)   # LeNet, Alexnet VGG16
parser.add_argument("data", help="Dataset name", type=str)   # MNIST CIFAR10
parser.add_argument("--test", help="if test", dest='isTest', action='store_const', const=True, default=False)
args = parser.parse_args()
if args.net:
    net_name = args.net
else:
    net_name = 'VGG16'
if args.data:
    dataset_name = args.data
else:
    dataset_name = 'CIFAR10'

net_and_data = net_name + '_' + dataset_name
prune_path_root = './pruning/result/'
prune_result_path = prune_path_root + net_and_data + '_retrain'

retrain_codebook_root = './quantization/result/'
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)
retrain_codebook_name = net_and_data + '_codebook'
retrain_epoch = 1

use_cuda = torch.cuda.is_available()
train_batch_size = 1
test_batch_size = 32
parallel_gpu = False
lr_list = {
    'LeNet': 1e-4,
    'AlexNet': 1e-4,
    'VGG16': 1e-4
}
lr = lr_list[net_name]
prune_fc_bits = 4
quantization_conv_bits = 8
quantization_fc_bits = 4
data_dir = './data'
retrain_codebook_path = retrain_codebook_root + retrain_codebook_name
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)


if net_name == 'LeNet':
    prune_net = PruneLeNet5()
elif net_name == 'AlexNet':
    if dataset_name == 'CIFAR10':
        prune_net = PruneAlexNet(num_classes=10)
    elif dataset_name == 'CIFAR100':
        prune_net = PruneAlexNet(num_classes=100)
elif net_name == 'VGG16':
    if dataset_name == 'CIFAR10':
        prune_net = PruneVGG16(num_classes=10)
    elif dataset_name == 'CIFAR100':
        prune_net = PruneVGG16(num_classes=100)
else:
    prune_net = None

if net_name == 'LeNet':
    net = LeNet5()
elif net_name == 'AlexNet':
    if dataset_name == 'CIFAR10':
        net = AlexNet(num_classes=10)
    elif dataset_name == 'CIFAR100':
        net = AlexNet(num_classes=100)
elif net_name == 'VGG16':
    if dataset_name == 'CIFAR10':
        net = VGG16(num_classes=10)
    elif dataset_name == 'CIFAR100':
        net = VGG16(num_classes=100)
else:
    net = None

if args.isTest:
    helper.sparse_to_init(net, prune_result_path, prune_fc_bits)
else:
    conv_layer_length, codebook, nz_num, conv_diff, fc_diff = share_weight(
        prune_net, prune_result_path, quantization_conv_bits, quantization_fc_bits, prune_fc_bits)
        
    max_value = np.finfo(np.float32).max
    max_conv_bit = 2 ** quantization_conv_bits
    max_fc_bit = 2 ** quantization_fc_bits
    index_list, key_parameter = helper.codebook_to_init(
        net, conv_layer_length, nz_num, conv_diff, fc_diff, codebook, max_conv_bit, max_fc_bit)
        
num_workers_list = {
    'LeNet': 16,
    'AlexNet': 16,
    'VGG16': 32
}
num_workers = num_workers_list[net_name]
trainloader, testloader = load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers, name=dataset_name, net_name=net_name)


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

print('Test')
test(use_cuda, testloader, net)
if args.isTest:
    os._exit(0)
print('Begin fine tune')
helper.train_codebook(key_parameter, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length, codebook,
                      index_list, testloader, net, trainloader, criterion,
                      optimizer, retrain_epoch)

helper.save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, retrain_codebook_path, net)

log.log_file_size(retrain_codebook_path, 'K')
