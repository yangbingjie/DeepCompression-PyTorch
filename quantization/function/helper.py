import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np


def sparse_to_init(net, conv_layer_length, sparse_conv_diff, sparse_fc_diff, conv_bits, fc_bits, codebook):
    state_dict = net.state_dict()
    conv_layer_index = 0
    fc_layer_index = 0
    for i, (key, value) in enumerate(state_dict.items()):
        # print(key, value.shape, codebook.conv_codebook_index, codebook.conv_codebook_value)
        shape = value.shape
        # print(value.shape)
        value = value.view(-1)
        # print(value.shape)
        value.zero_()
        layer_num = torch.numel(value)
        if i < conv_layer_length:
            layer_diff = sparse_conv_diff[conv_layer_index:conv_layer_index + layer_num]
            conv_layer_index += layer_num
        else:
            layer_diff = sparse_fc_diff[fc_layer_index:fc_layer_index + layer_num]
            fc_layer_index += layer_num
        dense_index = 0
        sparse_index = 0
        while sparse_index < layer_num:
            dense_index += layer_diff[sparse_index]
            # print(dense_index, sparse_index)
            value[0] = 2
            print(value[0].shape)
            print(codebook.codebook_value[i][codebook.codebook_index[i][sparse_index]])
            value[dense_index] = torch.from_numpy(codebook.codebook_value[i][codebook.codebook_index[i][sparse_index]])
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)


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
