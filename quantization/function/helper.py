import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np
import time
from tqdm import tqdm


def restructure_index(index_list, conv_layer_length, max_conv_bit, max_fc_bit):
    new_index_list = []
    count_list = []
    for i in range(len(index_list)):
        num = max_conv_bit if i < conv_layer_length else max_fc_bit
        temp = list(range(num))
        for j in range(len(index_list[i])):
            temp[index_list[i][j]].append(j)
        new_index_list.append(temp)
        count_list.append(len(temp))
    count_list = count_list[0::2] + count_list[1::2]
    return new_index_list, count_list


def sparse_to_init(net, conv_layer_length, nz_num, sparse_conv_diff, sparse_fc_diff, codebook, max_conv_bit,
                   max_fc_bit):
    state_dict = net.state_dict()
    index_list = []
    conv_layer_index = 0
    fc_layer_index = 0
    for i, (key, value) in enumerate(state_dict.items()):
        # print(key, value.shape, codebook.conv_codebook_index, codebook.conv_codebook_value)
        shape = value.shape
        # print(value.shape)
        value = value.view(-1)

        index = np.empty_like(value, dtype=np.uint8)
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
    new_index_list, count_list = restructure_index(index_list, conv_layer_length, max_conv_bit, max_fc_bit)
    return new_index_list, count_list


# def compute_cluster_count(index_list, conv_layer_length, max_conv_bit, max_fc_bit):
#     half_length = int(len(index_list) / 2)
#     cluster_count = []
#     for i in range(half_length):
#         cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
#         temp = np.empty(cluster_bits, dtype=np.uint8)
#         for j in range(cluster_bits):
#             temp[j] = len(index_list[i][index_list[i] == j]) + len(index_list[i + 1][index_list[i + 1] == j])
#         cluster_count.append(temp)
#     return cluster_count


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


def update_codebook(count_list, codebook, net, index_list, max_conv_bit, max_fc_bit, conv_layer_length):
    params = list(net.parameters())
    # print('========Start========')
    for i in range(0, len(params), 2):
        # start = time.clock()
        para = params[i]
        grad_shape = para.grad.shape
        grad = para.grad
        grad = grad.view(-1)
        index = index_list[i]

        bias = params[i + 1]
        bias_grad_shape = bias.grad.shape
        bias_grad = bias.grad
        bias_grad = bias_grad.view(-1)
        bias_index = index_list[i + 1]

        half_index = int(i / 2)
        # Cluster grad using index, use mean of each class of grad to update codebook centroids and update weight
        # Update codebook centroids
        cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
        codebook_centroids = codebook.codebook_value[half_index]

        # elapsed = (time.clock() - start)
        # print(round(elapsed, 5))

        start = time.clock()
        for j in range(cluster_bits):
            temp = grad[index]
            sum_grad = temp.sum()

            bias_temp = bias_grad[bias_index]
            sum_grad += bias_temp.sum()

            mean_grad = sum_grad / count_list[half_index]

            codebook_centroids[j] += mean_grad
            grad[index] = mean_grad
            bias_grad[bias_index] = mean_grad

        elapsed = (time.clock() - start)
        print(round(elapsed, 5))

        # start = time.clock()
        grad = grad.view(grad_shape)
        params[i].grad = grad.clone()

        bias_grad = bias_grad.view(bias_grad_shape)
        params[i + 1].grad = bias_grad.clone()

        # elapsed = (time.clock() - start)
        # print(round(elapsed, 5))
    # print('=========End=========')


def train_codebook(count_list, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length,
                   codebook, index_list, testloader, net, trainloader, criterion, optimizer,
                   train_path, epoch=1, accuracy_accept=99, epoch_step=25):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    # max_accuracy = 0
    for epoch in range(epoch):  # loop over the dataset multiple times
        start = time.clock()
        train_loss = []
        net.train()
        for inputs, labels in tqdm(trainloader):
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

            update_codebook(count_list, codebook, net, index_list, max_conv_bit, max_fc_bit, conv_layer_length)

            optimizer.step()  # update weight

            train_loss.append(loss.item())

        elapsed = (time.clock() - start)
        print(epoch, round(elapsed, 5))
        print('=========End=========')

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        accuracy = test(testloader, net, use_cuda)
        scheduler.step()
        # if accuracy > max_accuracy:
        #     torch.save(net.state_dict(), train_path)
        #     max_accuracy = accuracy
        # if accuracy > accuracy_accept:
        #     break
