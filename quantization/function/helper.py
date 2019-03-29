import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np


def sparse_to_init(net, conv_layer_length, codebook, nz_num, conv_diff, fc_diff):
    state_dict = net.state_dict()
    layer_index = 0
    for (key, value) in state_dict.items():
        if layer_index < conv_layer_length:
            state_dict[key] = ''


def test(testloader, net):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    return accuracy


def train_codebook(testloader, net, trainloader, criterion, optimizer, train_path, epoch=1,
                   accuracy_accept=99, epoch_step=25):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    max_accuracy = 0
    for epoch in range(epoch):  # loop over the dataset multiple times
        train_loss = []
        net.train()
        for inputs, labels in trainloader:
            # get the inputs
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)  # forward
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backward
            optimizer.step()  # update weight

            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        accuracy = test(testloader, net)
        scheduler.step()
        if accuracy > max_accuracy:
            torch.save(net.state_dict(), train_path)
            max_accuracy = accuracy
        if accuracy > accuracy_accept:
            break
