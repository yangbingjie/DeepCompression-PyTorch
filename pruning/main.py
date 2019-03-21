import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
import pruning.function.helper as helper
import util.log as log
import torch.optim as optim
import torch.multiprocessing as multiprocessing

# # test csr
# a = np.array([0, 3.4, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.7])
# tensor = torch.from_numpy(a)
# csr_matrix = WeightCSR(tensor, index_bits=3)
# a, b = csr_matrix.tensor_to_csr()
# print(a)
# print(b)
# print(bin(8)[2:].zfill(3))


use_cuda = True
seed = 46
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
multiprocessing.set_start_method('spawn')
torch.cuda.manual_seed(42)
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_path = './pruning/result/LeNet'
retrain_path = './pruning/result/LeNet_retrain'
retrain_num = 8
train_epoch = 128
retrain_epoch = 8
net = LeNet5().to(device)
lr = 1e-2
# weight_decay is L2 regularization
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
if os.path.exists(train_path):
    net.load_state_dict(torch.load(train_path, map_location='cpu' if not use_cuda else 'gpu'))
helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epoch=train_epoch)
torch.save(net.state_dict(), train_path)
log.log_file_size(train_path, 'K')
helper.test(testloader, net)
net.load_state_dict(torch.load(train_path))
net.eval()
for j in range(retrain_num):
    retrain_mode = 'conv' if j % 2 == 0 else 'fc'
    net.prune_layer(prune_mode=retrain_mode)
    print('====================== Retrain', retrain_mode, j, 'Start ==================')
    net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    # After pruning, the network is retrained with 1/10 of the original network's learning rate
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr / 10, weight_decay=1e-5)
    helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer,
                 epoch=retrain_epoch)
    helper.save_sparse_model(net, retrain_path)
    log.log_file_size(retrain_path, 'K')
    print('====================== ReTrain End ======================')

# prune rate:  5.80908
# The file size is 400.81 K
