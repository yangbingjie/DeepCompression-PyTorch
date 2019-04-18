import os
import torch
import argparse
import torch.nn as nn
import util.log as log
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pruning.function.helper as helper
from pruning.net.PruneVGG16 import PruneVGG16
from pruning.net.PruneLeNet5 import PruneLeNet5
import torch.optim.lr_scheduler as lr_scheduler
from pruning.net.PruneAlexNet import PruneAlexNet

parser = argparse.ArgumentParser()
parser.add_argument("net", help="Network name", type=str)  # LeNet, AlexNet VGG16
parser.add_argument("data", help="Dataset name", type=str)  # MNIST CIFAR10
args = parser.parse_args()
if args.net:
    net_name = args.net
else:
    net_name = 'VGG16'
if args.data:
    dataset_name = args.data
else:
    dataset_name = 'CIFAR10'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using CUDA')
train_epoch_list = {
    'LeNet': 4,
    'AlexNet': 100,
    'VGG16': 200,
}
train_epoch = train_epoch_list[net_name]
# Prune sensitivity
sensitivity_list = {
    'LeNet': {
        'conv1': 0.39,
        'conv2': 0.8,
        'fc1': 1.05,  # 33687 1 27665 1.1
        'fc2': 0.76
    },
    'AlexNet': {
        'conv1': 0.3,
        'conv': 0.5,
        'fc': 0.77,
    },
    'VGG16': {
        'conv1': 0.31,
        'conv2': 0.57,
        'conv3': 0.53,
        'conv4': 0.55,
        'conv5': 0.45,
        'conv6': 0.655,
        'conv7': 0.51,
        'conv8': 0.59,
        'conv9': 0.63,
        'conv10': 0.59,
        'conv11': 0.59,
        'conv12': 0.64,
        'conv13': 0.555,
        'fc1': 1.1,  # 3次之后 90.71 1.2
        'fc2': 1.1,
        'fc3': 0.64,
    }
}
sensitivity = sensitivity_list[net_name]
print(sensitivity)
# When accuracy in test dataset is more than max_accuracy, save the model
max_accuracy_list = {
    'LeNet': 99.3,
    'AlexNet': 90,
    'VGG16': 93
}
max_accuracy = max_accuracy_list[net_name]
# LeNet: 330 3000 32000 950
retrain_mode_list = {
    'LeNet': [
        {'mode': 'full', 'retrain_epoch': 10},
        {'mode': 'full', 'retrain_epoch': 20},
        {'mode': 'full', 'retrain_epoch': 20}
    ],

    'AlexNet': [
        {'mode': 'conv', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
    ],
    'VGG16': [
        {'mode': 'fc', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
        {'mode': 'fc', 'retrain_epoch': 20},
        # {'mode': 'conv', 'retrain_epoch': 10},
        # {'mode': 'conv', 'retrain_epoch': 10},
        # {'mode': 'conv', 'retrain_epoch': 10},
        # {'mode': 'conv', 'retrain_epoch': 10},
        # {'mode': 'conv', 'retrain_epoch': 10}
    ]
}

retrain_mode_type = retrain_mode_list[net_name]
print(retrain_mode_type)

learning_rate_decay_list = {
    'LeNet': 1e-5,
    'AlexNet': 1e-5,
    'VGG16': 5e-4
}
learning_rate_decay = learning_rate_decay_list[net_name]
prune_num_per_retrain = 3
train_batch_size_list = {
    'LeNet': 32,
    'AlexNet': 64,
    'VGG16': 128
}
train_batch_size = train_batch_size_list[net_name]

test_batch_size = 64

lr_list = {
    'LeNet': 1e-2,
    'AlexNet': 1e-2,
    'VGG16': 1e-2
}
lr = lr_list[net_name]

train_milestones_list = {
    'LeNet': [],
    'AlexNet': [30, 50],
    'VGG16': [100, 180]
}
# After pruning, the LeNet is retrained with 1/10 of the original network's learning rate
# After pruning, the AlexNet is retrained with 1/100 of the original network's learning rate
retrain_lr_list = {
    'LeNet': lr / 10,
    'AlexNet': lr / 100,
    'VGG16': lr / 100
}
retrain_lr = retrain_lr_list[net_name]
retrain_milestones_list = {
    'LeNet': [],
    'AlexNet': [20, 30],
    'VGG16': [15]
}

if net_name == 'LeNet':
    net = PruneLeNet5()
elif net_name == 'AlexNet':
    net = PruneAlexNet(num_classes=10)
elif net_name == 'VGG16':
    net = PruneVGG16(num_classes=10)
else:
    net = None

path_root = './pruning/result/'
train_path = path_root + net_name
retrain_path = train_path + '_retrain'
num_workers_list = {
    'LeNet': 16,
    'AlexNet': 16,
    'VGG16': 32
}
num_workers = num_workers_list[net_name]
trainloader, testloader = helper.load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers,
                                              name=dataset_name)

if not os.path.exists(path_root):
    os.mkdir(path_root)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=learning_rate_decay)
train_milestones = train_milestones_list[net_name]
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=train_milestones, gamma=0.1)

if use_cuda:
    # move param and buffer to GPU
    net = net.cuda()
    # speed up slightly
    cudnn.benchmark = True

# weight_decay is L2 regularization
if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path))
else:
    helper.train(testloader, net, trainloader, criterion, optimizer, train_path, scheduler, max_accuracy,
                 epoch=train_epoch, use_cuda=use_cuda)
    torch.save(net.state_dict(), train_path)
if net_name == 'LeNet':
    unit = 'K'
else:
    unit = 'M'
log.log_file_size(train_path, unit)
helper.test(use_cuda, testloader, net)

# Retrain
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=retrain_lr,
                      weight_decay=learning_rate_decay)

retrain_milestones = retrain_milestones_list[net_name]
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=retrain_milestones, gamma=0.1)
for j in range(len(retrain_mode_type)):
    retrain_mode = retrain_mode_type[j]['mode']
    net.prune_layer(prune_mode=retrain_mode, use_cuda=use_cuda, sensitivity=sensitivity)
    if hasattr(net, 'drop_rate') and retrain_mode == 'fc':
        net.compute_dropout_rate()
    print('====================== Retrain', j, retrain_mode, 'Start ==================')
    if retrain_mode_type[j]['mode'] != 'full':
        net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    helper.train(testloader, net, trainloader, criterion, optimizer, retrain_path, scheduler, max_accuracy,
                 unit, use_cuda=use_cuda, epoch=retrain_mode_type[j]['retrain_epoch'], save_sparse=True)
    print('====================== ReTrain End ======================')

helper.save_sparse_model(net, retrain_path, unit)
