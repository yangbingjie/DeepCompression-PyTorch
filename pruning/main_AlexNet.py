import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.AlexNet import AlexNet
from pruning.function.helper import train, test
import torchvision.datasets as datasets
import util.log as log
import torch.optim as optim

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

trainset = datasets.ImageFolder(root='../data/tiny-imagenet-200/train',
                                transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = datasets.ImageFolder(root='../data/tiny-imagenet-200/val',
                               transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss()

retrain_num = 2
train_path = '../pruning/result/AlexNet'
retrain_path = '../pruning/result/AlexNet_retrain'
net = AlexNet(num_classes=200)
lr = 1e-2
optimizer = torch.optim.SGD(list(net.parameters())[:], lr=lr, momentum=0.9, weight_decay=1e-5)
train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
torch.save(net.state_dict(), train_path)
log.log_file_size(train_path, 'M')
test(testloader, net)

for j in range(retrain_num):
    retrain_mode = 'conv' if j % 2 == 0 else 'fc'
    net.load_state_dict(torch.load(retrain_path if j != 0 else train_path))
    net.eval()
    net.prune_layer(prune_mode=retrain_mode)
    print('====================== Retrain', retrain_mode, j, 'Start ==================')
    net.compute_dropout_rate()
    net.fix_layer(fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr / 100, momentum=0.9,
                                weight_decay=1e-5)
    train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
    torch.save(net.state_dict(), retrain_path)
    log.log_file_size(retrain_path, 'M')
    test(testloader, net)
    print('====================== ReTrain End ======================')
