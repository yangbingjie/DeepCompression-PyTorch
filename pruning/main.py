import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
from pruning.function.helper import train, test
import util.log as log

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

criterion = nn.CrossEntropyLoss()

retrain_num = 3
base_path = './pruning/result/LeNet'
net = LeNet5()
train(net, trainloader=trainloader, criterion=criterion)
path = base_path + '0'
torch.save(net.state_dict(), path)
log.log_file_size(path, 'K')
test(testloader, net)

for j in range(retrain_num):
    print('=========== Retrain', j, ' Start ===========')
    net.load_state_dict(torch.load(base_path + str(j)))
    net.eval()
    train(net, trainloader=trainloader, criterion=criterion)
    net.prune_layer()
    path = base_path + str(j + 1)
    torch.save(net.state_dict(), path)
    log.log_file_size(path, 'K')
    print('=========== Train End ===========')
    test(testloader, net)



