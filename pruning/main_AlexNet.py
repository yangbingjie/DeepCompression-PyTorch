import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
from pruning.function.helper import train, test
import util.log as log

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

trainset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/train', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='../data/tiny-imagenet-200/test', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

criterion = nn.CrossEntropyLoss()

retrain_num = 3
path = './pruning/result/AlexNet'
net = LeNet5()
train(net, trainloader=trainloader, criterion=criterion)
torch.save(net.state_dict(), path + '0')
log.log_file_size(path, 'K')
test(testloader, net)

for j in range(retrain_num):
    print('=========== Retrain Start ===========')
    net.load_state_dict(torch.load(path + str(retrain_num - 1)))
    net.eval()
    net.compute_dropout_rate()
    train(net, trainloader=trainloader, criterion=criterion)
    net.prune_layer()
    path = path + str(j)
    torch.save(net.state_dict(), path)
    log.log_file_size(path, 'K')
    print('=========== Train End ===========')
    test(testloader, net)



