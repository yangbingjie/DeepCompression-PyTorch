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
import torch.optim.lr_scheduler as lr_scheduler
from pruning.function.helper import load_dataset
from quantization.function.weight_share import share_weight

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("net", help="Network name", type=str)  # LeNet, Alexnet VGG16
parser.add_argument("data", help="Dataset name", type=str)  # MNIST CIFAR10 CIFAR100
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
prune_result_path = prune_path_root + net_and_data + '_retrain.pth'

retrain_codebook_root = './quantization/result/'
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)
retrain_codebook_name = net_and_data + '_codebook'
retrain_epoch = 15
learning_rate_decay_list = {
    'LeNet': 1e-5,
    'AlexNet': 1e-3,
    'VGG16': 5e-4,
}
learning_rate_decay = learning_rate_decay_list[net_name]
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('USING CUDA')
else:
    print('USING CPU')
train_batch_size = 1
test_batch_size = 32
parallel_gpu = False
lr_list = {
    'LeNet': 1e-5,
    'AlexNet': 1e-5,
    'VGG16': 1e-6 * 5
}
lr = lr_list[net_name]
prune_fc_bits = 4
quantization_conv_bits = 8
quantization_fc_bits = 4
data_dir = './data'
retrain_codebook_path = retrain_codebook_root + retrain_codebook_name + '.pth'
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)

if dataset_name == 'CIFAR100':
    top_5 = True
else:
    top_5 = False
prune_net = None
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

net = None
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

num_workers_list = {
    'LeNet': 16,
    'AlexNet': 32,
    'VGG16': 32
}
num_workers = num_workers_list[net_name]
trainloader, testloader = load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers, name=dataset_name,
                                       net_name=net_name)

if args.isTest:
    helper.sparse_to_init(net, prune_result_path, prune_fc_bits)
    if use_cuda:
        # move param and buffer to GPU
        net = net.cuda()
        if parallel_gpu:
            # parallel use GPU
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
        # speed up slightly
        cudnn.benchmark = True
    print('Test')
    test(use_cuda, testloader, net, top_5)
else:
    conv_layer_length, codebook, nz_num, conv_diff, fc_diff = share_weight(
        prune_net, prune_result_path, quantization_conv_bits, quantization_fc_bits, prune_fc_bits)

    max_value = np.finfo(np.float32).max
    max_conv_bit = 2 ** quantization_conv_bits
    max_fc_bit = 2 ** quantization_fc_bits
    index_list, key_parameter = helper.codebook_to_init(
        net, conv_layer_length, nz_num, conv_diff, fc_diff, codebook, max_conv_bit, max_fc_bit)
    if use_cuda:
        # move param and buffer to GPU
        net = net.cuda()
        if parallel_gpu:
            # parallel use GPU
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
        # speed up slightly
        cudnn.benchmark = True
    print('Test')
    test(use_cuda, testloader, net, top_5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=learning_rate_decay)
    retrain_milestones_list = {
        'LeNet': [2],
        'AlexNet': [5],
        'VGG16': [5]
    }
    retrain_milestones = retrain_milestones_list[net_name]
    print('Begin fine tune')
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=retrain_milestones, gamma=0.1)
    max_accuracy_list = {
        'LeNet': 99,
        'AlexNet': 88.82,
        'VGG16': 90.39
    }
    max_accuracy = max_accuracy_list[net_name]
    helper.train_codebook(max_accuracy, nz_num, conv_diff, fc_diff, retrain_codebook_path,
                          key_parameter, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length,
                          codebook, index_list, testloader, net, trainloader, criterion,
                          optimizer, scheduler, retrain_epoch, top_5)

    helper.save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, retrain_codebook_path, net)

    log.log_file_size(retrain_codebook_path, 'K')
