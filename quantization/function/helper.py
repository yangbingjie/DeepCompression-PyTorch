import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np
import time
import math
from tqdm import tqdm


def load_sparse_model(net, path, bits):
    conv_layer_num = 0
    fc_layer_num = 0
    fin = open(path, 'rb')
    for name, x in net.named_parameters():
        if name.endswith('mask'):
            continue
        if name.startswith('conv'):
            conv_layer_num += 1
        elif name.startswith('fc'):
            fc_layer_num += 1
    nz_num = np.fromfile(fin, dtype=np.uint32, count=conv_layer_num + fc_layer_num)

    conv_diff_num = sum(nz_num[:conv_layer_num])
    conv_diff = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)

    fc_merge_num = math.floor((sum(nz_num[conv_layer_num:]) + 1) / 2)
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)

    conv_value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num[:conv_layer_num]))
    fc_value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num[conv_layer_num:]))

    # print(nz_num)
    # print(conv_diff.size, conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # print(conv_value_array.size, conv_value_array[-10:])
    # print(fc_value_array.size, fc_value_array[-10:])

    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 39316 [ 17 242  34  50 164  44  26   3   6 128]
    # 8537 [ 0.05500366 -0.0518913  -0.05787839  0.04747333 -0.07086759 -0.07142863
    #  -0.06043605 -0.06711546 -0.0698091  -0.06924898]
    # 78631 [ 0.13233908  0.16305041 -0.171971   -0.1353672   0.16033891 -0.19598335
    #  -0.11460102 -0.32042998 -0.12170218  0.14367148]

    # Split 8 bits index to 4 bits index
    fc_diff = []
    max_bits = 2 ** bits
    for i in range(len(fc_merge_diff)):
        fc_diff.append(int(fc_merge_diff[i] / max_bits))  # first 4 bits
        fc_diff.append(fc_merge_diff[i] % max_bits)  # last 4 bits
    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0:
        fc_diff = fc_diff[:fc_num_sum]
    fc_diff = np.asarray(fc_diff, dtype=np.uint8)

    # layer_index = fc_diff[0:0 + nz_num[4]]
    # print(sum(layer_index) + len(layer_index))

    # print(nz_num)
    # print(conv_diff.size, conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # print(conv_value_array.size, conv_value_array[-10:])
    # print(fc_value_array.size, fc_value_array[-10:])

    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 78631 [ 4  2 12  1 10  0  3  0  6  8]
    # 8537 [ 0.05500366 -0.0518913  -0.05787839  0.04747333 -0.07086759 -0.07142863
    #  -0.06043605 -0.06711546 -0.0698091  -0.06924898]
    # 78631 [ 0.13233908  0.16305041 -0.171971   -0.1353672   0.16033891 -0.19598335
    #  -0.11460102 -0.32042998 -0.12170218  0.14367148]

    return conv_layer_num, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array


def restructure_index(index_list, conv_layer_length, max_conv_bit, max_fc_bit):
    new_index_list = []
    new_count_list = []
    count_list = []

    for i in range(len(index_list)):
        num = max_conv_bit if i < conv_layer_length else max_fc_bit
        tmp_index = []
        tmp_count = []
        for j in range(num):
            tmp_index.append(np.where(np.array(index_list[i]) == j)[0].tolist())
            tmp_count.append(len(tmp_index[j]))
        new_index_list.append(tmp_index)
        count_list.append(tmp_count)

    for k in range(0, len(count_list), 2):
        new_count_list.append(np.sum([count_list[k], count_list[k + 1]], axis=0).tolist())

    return new_index_list, new_count_list


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
        codebook_index_array = codebook.codebook_index[half_index]
        # print(layer_diff.sum() + len(layer_diff))
        while sparse_index < len(layer_diff):
            dense_index += layer_diff[sparse_index]
            # if dense_index == 400000:
            # print(sparse_index)
            value[dense_index] = float(codebook.codebook_value[half_index][codebook_index_array[sparse_index]])
            index[dense_index] = int(codebook_index_array[sparse_index])
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
        #
        # start = time.clock()
        for j in range(cluster_bits):
            sum_grad = grad[index[j]].sum()

            sum_grad += bias_grad[bias_index[j]].sum()

            mean_grad = sum_grad / count_list[half_index][j]

            codebook_centroids[j] += mean_grad
            grad[index[j]] = mean_grad
            bias_grad[bias_index[j]] = mean_grad

        # elapsed = (time.clock() - start)
        # print(round(elapsed, 5))
        #
        # start = time.clock()
        grad = grad.view(grad_shape)
        params[i].grad = grad.clone()

        bias_grad = bias_grad.view(bias_grad_shape)
        params[i + 1].grad = bias_grad.clone()

    #     elapsed = (time.clock() - start)
    #     print(round(elapsed, 5))
    # print('=========End=========')


def train_codebook(count_list, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length,
                   codebook, index_list, testloader, net, trainloader, criterion, optimizer,
                   train_path, epoch=1, accuracy_accept=99, epoch_step=25):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    # max_accuracy = 0
    for epoch in range(epoch):  # loop over the dataset multiple times
        # start = time.clock()
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

            # TODO delete
            break

        # elapsed = (time.clock() - start)
        # print(epoch, round(elapsed, 5))
        # print('=========End=========')

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        accuracy = test(testloader, net, use_cuda)
        scheduler.step()

        # # TODO delete
        # break

        # if accuracy > max_accuracy:
        #     torch.save(net.state_dict(), train_path)
        #     max_accuracy = accuracy
        # if accuracy > accuracy_accept:
        #     break


def save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, path):
    fc_merge_diff = []

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 78631 [ 4  2 12  1 10  0  3  0  6  8]

    length = len(fc_diff)
    fc_diff = list(fc_diff)
    if length % 2 != 0:
        fc_diff.append(0)
    for i in range(math.floor(len(fc_diff) / 2)):
        fc_merge_diff.append((fc_diff[2 * i] << 4) | fc_diff[2 * i + 1])
    nz_num = np.asarray(nz_num, dtype=np.uint32)
    conv_diff = np.asarray(conv_diff, dtype=np.uint8)
    fc_merge_diff = np.asarray(fc_merge_diff, dtype=np.uint8)

    conv_half_len = int(conv_layer_length / 2)
    conv_codebook_index = []
    for m in range(conv_half_len):
        conv_codebook_index.extend(codebook.codebook_index[m])

    fc_codebook_index = []
    for k in range(conv_half_len, len(codebook.codebook_index)):
        fc_codebook_index.extend(codebook.codebook_index[k])

    codebook_value = []
    for j in range(len(codebook.codebook_value)):
        codebook_value.extend(codebook.codebook_value[j])

    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index), fc_codebook_index[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # 8537 [136, 122, 119, 132, 236, 73, 126, 75, 16, 74]
    # 78631 [7, 6, 3, 9, 6, 2, 5, 0, 5, 11]
    # 544 [0.15731171, 0.11615839, -0.00030401052, -0.12842683, 0.12538931, 0.122185014, 0.18652469, 0.19523832, 0.25232622, 0.296758]
    length = len(fc_codebook_index)
    if length % 2 != 0:
        fc_codebook_index.append(0)

    fc_codebook_index = np.array(fc_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = []
    for i in range(math.floor((len(fc_codebook_index)) / 2)):
        fc_codebook_index_merge.append(
            (fc_codebook_index[2 * i] << 4) | fc_codebook_index[2 * i + 1])

    conv_codebook_index = np.asarray(conv_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = np.asarray(fc_codebook_index_merge, dtype=np.uint8)
    codebook_value = np.asarray(codebook_value, dtype=np.float32)

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index_merge), fc_codebook_index_merge[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 39316 [ 17 242  34  50 164  44  26   3   6 128]
    # 8537 [136 122 119 132 236  73 126  75  16  74]
    # 39316 [170 134 100 198 167  99 150  37   5 176]
    # 544 [ 0.15731171  0.11615839 -0.00030401 -0.12842683  0.12538931  0.12218501
    #   0.18652469  0.19523832  0.25232622  0.296758  ]

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    codebook_value.dtype = np.uint8

    sparse_obj = np.concatenate((nz_num, conv_diff, fc_merge_diff, conv_codebook_index,
                                 fc_codebook_index_merge, codebook_value))
    sparse_obj.tofile(path)
