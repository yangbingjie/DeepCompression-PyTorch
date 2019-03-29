import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
from quantization.function.weight_share import share_weight
from quantization.net.LookupLeNet5 import LookupLeNet5
import quantization.function.helper as helper
import torch.optim as optim

prune_result_path = './pruning/result/LeNet_retrain'
retrain_codebook_root = './quantization/result/'
retrain_codebook_name = 'LeNet_retrain'
retrain_epoch = 1

use_cuda = torch.cuda.is_available()
train_batch_size = 16
test_batch_size = 16
loss_accept = 1e-2
lr = 1e-2
retrain_num = 2
train_epoch = 1

data_dir = './data'
retrain_codebook_path = retrain_codebook_root + retrain_codebook_name
if not os.path.exists(retrain_codebook_root):
    os.mkdir(retrain_codebook_root)

prune_net = LeNet5()
conv_layer_length, codebook, nz_num, conv_diff, fc_diff = share_weight(prune_net, prune_result_path, 8, 5)
helper.sparse_to_init(prune_net, conv_layer_length, codebook, nz_num, conv_diff, fc_diff)

#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize([0.5], [0.5])])
# # Loader
# kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
#
# trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
#                                       download=True, transform=transform)
#
# testset = torchvision.datasets.MNIST(root=data_dir, train=False,
#                                      download=True, transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
#                                           shuffle=True,
#                                           **kwargs)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
#                                          **kwargs)
#
# net = LookupLeNet5()
# net.load_state_dict(state_dict)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
#
# # helper.train_codebook(testloader, sparse_net, trainloader, criterion, optimizer, retrain_codebook_path, retrain_epoch)
