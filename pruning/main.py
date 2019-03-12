import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pruning.net.LeNet5 import LeNet5
import util.log as log


def train(is_retrain=True, retrain_num=0, path='./model/Untitled'):
    net = LeNet5()
    if is_retrain:
        print('=========== Retrain:', retrain_num, ' =========')
        net.load_state_dict(torch.load(path + str(retrain_num)))
        net.eval()
    else:
        print('=========== Train Start ===========')


    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # backward
            optimizer.step()  # update weight

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                break

    path = path + str(retrain_num + 1)
    torch.save(net.state_dict(), path)
    log.log_file_size(path, 'K')
    print('=========== Train End ===========')


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

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

criterion = nn.CrossEntropyLoss()

retrain_num = 3
path = './model/LeNet'

train(is_retrain=False, path=path)

for j in range(retrain_num):
    train(retrain_num=j, path=path)



