import os
import torch
import torch.nn as nn
import util.log as log
import torch.optim as optim
from pruning.net.VGG16 import VGG16
import torch.backends.cudnn as cudnn
from pruning.net.AlexNet import AlexNet
import pruning.function.helper as helper
from pruning.net.PruneLeNet5 import PruneLeNet5

net_type = 'LeNet'  # Alexnet VGG16
data_type = 'MNIST'  # CIFAR10

device_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parallel_gpu = False
use_cuda = torch.cuda.is_available()
train_epoch_list = {
    'LeNet': 4,
    'AlexNet': 100,
    'VGG16': 200,
}
train_epoch = train_epoch_list[net_type]
sensitivity_list = {
    'LeNet': {
        'conv1': 0.4,
        'conv2': 0.73,
        'fc1': 0.9,
        'fc2': 0.7
    },
    'AlexNet': {
        'conv1': 0.3,
        'conv': 0.5,
        'fc': 0.77,
    },
    'VGG16': {
        'conv1': 0.5,
        'conv2': 0.6,
        'conv5': 0.7,
        'fc1': 0.99,
        'conv': 0.8,
        'fc': 0.95,
    }
}
sensitivity = sensitivity_list[net_type]
print(sensitivity)

# LeNet: 330 3000 32000 950
retrain_mode_list = {
    'LeNet': [{'mode': 'full', 'prune_num': 1, 'retrain_epoch': 8}] * 5,
    'AlexNet': [
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8}
    ],
    'VGG16': [
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'conv', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8},
        {'mode': 'fc', 'prune_num': 1, 'retrain_epoch': 8}
    ]
}

retrain_mode = retrain_mode_list[net_type]
print(retrain_mode)

learning_rate_decay_list = {
    'LeNet': 1e-5,
    'AlexNet': 1e-5,
    'VGG16': 0.0005
}
learning_rate_decay = learning_rate_decay_list[net_type]
prune_num_per_retrain = 3
train_batch_size_list = {
    'LeNet': 32,
    'AlexNet': 64,
    'VGG16': 128
}
train_batch_size = train_batch_size_list[net_type]

test_batch_size = 64

lr = 1e-2
retrain_lr_list = {
    'LeNet': lr / 10,
    'AlexNet': lr / 100
}
retrain_lr = retrain_lr_list[net_type]

if net_type == 'LeNet':
    net = PruneLeNet5()
elif net_type == 'AlexNet':
    net = AlexNet(num_classes=10)
elif net_type == 'VGG16':
    net = VGG16(num_classes=10)
else:
    net = None

path_root = './pruning/result/'
train_path = path_root + net_type
retrain_path = train_path + '_retrain'
trainloader, testloader = helper.load_dataset(use_cuda, train_batch_size, test_batch_size, type=data_type)

if not os.path.exists(path_root):
    os.mkdir(path_root)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=learning_rate_decay)

if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    if parallel_gpu:
        # parallel use GPU
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() - 1))
    # speed up slightly
    cudnn.benchmark = True

# weight_decay is L2 regularization
if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path))
else:
    helper.train(testloader, net, trainloader, criterion, optimizer, train_path,
                 epoch=train_epoch, use_cuda=use_cuda, epoch_step=25)
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
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=retrain_lr,
                          weight_decay=learning_rate_decay)
    helper.train(testloader, net, trainloader, criterion, optimizer, retrain_path, use_cuda=use_cuda,
                 epoch=retrain_mode_list[j]['retrain_epoch'], save_sparse=True)
    print('====================== ReTrain End ======================')
    log.log_file_size(retrain_path, 'K')
