import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pruning.net.VGG16 import VGG16
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
base_path = './result/VGG'
net = VGG16(num_classes=200)
optimizer = optim.SGD(list(net.parameters())[:], lr=1e-3, momentum=0.9)
train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
path = base_path + '0'
torch.save(net.state_dict(), path)
log.log_file_size(path, 'M')
test(testloader, net)


for j in range(retrain_num):
    print('=========== Retrain Start ===========')
    net.load_state_dict(torch.load(base_path + str(j)))
    net.eval()
    net.prune_layer()
    net.compute_dropout_rate()
    train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
    path = base_path + str(j + 1)
    torch.save(net.state_dict(), path)
    log.log_file_size(path, 'M')
    print('=========== Train End ===========')
    test(testloader, net)
