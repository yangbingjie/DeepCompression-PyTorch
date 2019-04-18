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
import quantization.function.helper as helper
from pruning.net.PruneLeNet5 import PruneLeNet5
from pruning.function.helper import load_dataset
from quantization.function.weight_share import share_weight

parser = argparse.ArgumentParser()
parser.add_argument("net", help="Network name", type=str)   # LeNet, Alexnet VGG16
parser.add_argument("data", help="Dataset name", type=str)   # MNIST CIFAR10
args = parser.parse_args()
if args.net:
    net_name = args.net
else:
    net_name = 'VGG16'
if args.data:
    dataset_name = args.data
else:
    dataset_name = 'CIFAR10'

prune_path_root = './pruning/result/'
prune_result_path = prune_path_root + net_name + '_retrain'

retrain_codebook_root = './quantization/result/'
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)
retrain_codebook_name = net_name + '_codebook'
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
num_workers_list = {
    'LeNet': 16,
    'AlexNet': 16,
    'VGG16': 32
}
num_workers = num_workers_list[net_name]
trainloader, testloader = load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers, name=dataset_name)

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
