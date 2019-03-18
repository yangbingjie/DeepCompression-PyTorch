import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
import pruning.function.helper as helper
import util.log as log
import torch.optim as optim
from pruning.function.csr import WeightCSR

# # test csr
# a = np.array([0, 3.4, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.7])
# tensor = torch.from_numpy(a)
# csr_matrix = WeightCSR(tensor, index_bits=3)
# a, b = csr_matrix.tensor_to_csr()
# print(a)
# print(b)
# print(bin(8)[2:].zfill(3))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='../data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

retrain_num = 2
train_path = '../pruning/result/LeNet'
retrain_path = '../pruning/result/LeNet_retrain'
net = LeNet5()
lr = 1e-3
# weight_decay is L2 regularization
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5)
helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
torch.save(net.state_dict(), train_path)
log.log_file_size(train_path, 'K')
helper.test(testloader, net)

for j in range(retrain_num):
    retrain_mode = 'conv' if j % 2 == 0 else 'fc'
    net.eval()
    net.prune_layer(prune_mode=retrain_mode)
    print('====================== Retrain', retrain_mode, j, 'Start ==================')
    net.fix_layer(fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    # After pruning, the network is retrained with 1/10 of the original network's learning rate
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr / 10, weight_decay=1e-5)
    helper.train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
    print('====================== ReTrain End ======================')

helper.save_sparse_model(net, retrain_path)
log.log_file_size(retrain_path, 'K')
helper.test(testloader, net)
