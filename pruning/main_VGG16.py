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
train_path = '../pruning/result/VGG16'
retrain_path = './pruning/result/VGG16_retrain'
net = VGG16(num_classes=200)
lr = 1e-1
optimizer = optim.SGD(list(net.parameters())[:], lr=lr, momentum=0.9, weight_decay=1e-5)
train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
torch.save(net.state_dict(), train_path)
log.log_file_size(train_path, 'M')
test(testloader, net)


for j in range(retrain_num):
    print('=========== Retrain', j, 'Start ===========')
    net.load_state_dict(torch.load(retrain_path if j != 0 else train_path))
    net.eval()
    # We used five iterations of pruning an retraining
    for k in range(5):
        net.prune_layer()
    net.compute_dropout_rate()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr/100, momentum=0.9, weight_decay=1e-5)
    train(net, trainloader=trainloader, criterion=criterion, optimizer=optimizer)
    torch.save(net.state_dict(), retrain_path)
    log.log_file_size(retrain_path, 'M')
    test(testloader, net)
    print('=========== ReTrain End ===========')
