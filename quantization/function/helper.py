import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import numpy as np
import time


def sparse_to_init(net, conv_layer_length, nz_num, sparse_conv_diff, sparse_fc_diff, codebook):
    state_dict = net.state_dict()
    index_list = []
    conv_layer_index = 0
    fc_layer_index = 0
    for i, (key, value) in enumerate(state_dict.items()):
        # print(key, value.shape, codebook.conv_codebook_index, codebook.conv_codebook_value)
        shape = value.shape
        # print(value.shape)
        value = value.view(-1)

        index = torch.empty_like(value, dtype=torch.short)
        index[:] = -1
        # print(value.shape)
        value.zero_()
        if i < conv_layer_length:
            layer_diff = sparse_conv_diff[conv_layer_index:conv_layer_index + nz_num[i]]
            conv_layer_index += nz_num[i]
        else:
            layer_diff = sparse_fc_diff[fc_layer_index:fc_layer_index + nz_num[i]]
            fc_layer_index += nz_num[i]
        dense_index = 0
        sparse_index = 0
        half_index = int(i / 2)
        while sparse_index < len(layer_diff):
            dense_index += layer_diff[sparse_index]
            value[dense_index] = float(
                codebook.codebook_value[half_index][codebook.codebook_index[half_index][sparse_index]])
            index[dense_index] = int(codebook.codebook_index[half_index][sparse_index])
            # print(value[dense_index])
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)
        index.reshape(shape)
        index_list.append(index)
    # print(index_list)
    return index_list


def test(testloader, net, use_cuda):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = round(100 * correct / total, 2)
    print('Accuracy of the network on the test images: %f %%' % accuracy)
    return accuracy


def update_codebook(codebook, net, index_list, conv_bits, fc_bits, conv_layer_length):
    params = list(net.parameters())
    for i in range(0, len(params), 2):
        grad_shape = params[i].grad.shape
        grad = params[i].grad
        grad = grad.view(-1)
        index = index_list[i]

        bias_grad_shape = params[i + 1].grad.shape
        bias_grad = params[i + 1].grad
        bias_grad = bias_grad.view(-1)
        bias_index = index_list[i + 1]

        half_index = int(i / 2)

        # Cluster grad using index, use mean of each class of grad to update codebook centroids and update weight
        # print(index.shape)
        # print(grad.shape)

        # Update codebook centroids
        cluster_bits = conv_bits if i < conv_layer_length else fc_bits
        cluster_num = 2 ** cluster_bits
        codebook_centroids = codebook.codebook_value[half_index]
        for j in range(cluster_num):
            temp = grad[index == j]
            sum_grad = temp.sum()
            count = len(temp)

            bias_temp = bias_grad[bias_index == j]
            sum_grad += bias_temp.sum()
            count += len(bias_temp)

            mean_grad = sum_grad / count

            codebook_centroids[j] += mean_grad
            grad[index == j] = mean_grad
            bias_grad[bias_index == j] = mean_grad

        grad = grad.view(grad_shape)
        params[i].grad = grad.clone()

        bias_grad = bias_grad.view(bias_grad_shape)
        params[i + 1].grad = bias_grad.clone()


def train_codebook(use_cuda, conv_bits, fc_bits, conv_layer_length, codebook, index_list,
                   testloader, net, trainloader, criterion, optimizer, train_path,
                   epoch=1, accuracy_accept=99, epoch_step=25):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    # max_accuracy = 0
    for epoch in range(epoch):  # loop over the dataset multiple times
        train_loss = []
        net.train()
        start = time.clock()
        for inputs, labels in trainloader:
            # get the inputs
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)  # forward
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backward

            update_codebook(codebook, net, index_list, conv_bits, fc_bits, conv_layer_length)

            optimizer.step()  # update weight

            train_loss.append(loss.item())
        elapsed = (time.clock() - start)
        print('csr', round(elapsed, 5))
        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        accuracy = test(testloader, net, use_cuda)
        scheduler.step()
        # if accuracy > max_accuracy:
        #     torch.save(net.state_dict(), train_path)
        #     max_accuracy = accuracy
        # if accuracy > accuracy_accept:
        #     break
