import torch.optim.lr_scheduler as lr_scheduler
import torch
import numpy as np
import math
from tqdm import tqdm
from pruning.function.helper import test


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
            tmp_index.append(np.where(np.asarray(index_list[i]) == j)[0].tolist())
            tmp_count.append(len(tmp_index[j]))
        new_index_list.append(tmp_index)
        count_list.append(tmp_count)

    for k in range(0, len(count_list), 2):
        new_count_list.append(np.sum([count_list[k], count_list[k + 1]], axis=0).tolist())

    key_parameter = []
    for j in range(int(len(index_list) / 2)):
        layer_index = np.concatenate((index_list[2 * j], index_list[2 * j + 1]))
        num = max_conv_bit if j < (conv_layer_length / 2) else max_fc_bit
        empty_parameter = [None] * num
        key_parameter.append(empty_parameter)
        for m in range(len(layer_index)):
            # print(m, layer_index[m])
            if layer_index[m] != -1 and key_parameter[j][layer_index[m]] is None:
                key_parameter[j][layer_index[m]] = m
    return new_index_list, new_count_list, key_parameter


def sparse_to_init(net, conv_layer_length, nz_num, sparse_conv_diff, sparse_fc_diff, codebook, max_conv_bit,
                   max_fc_bit):
    state_dict = net.state_dict()
    index_list = []
    conv_layer_index = 0
    fc_layer_index = 0
    for i, (key, value) in enumerate(state_dict.items()):
        shape = value.shape
        value = value.view(-1)

        index = np.empty_like(value, dtype=np.int16)
        index[:] = -1
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
        while sparse_index < len(layer_diff):
            dense_index += layer_diff[sparse_index]
            value[dense_index] = float(codebook.codebook_value[half_index][codebook_index_array[sparse_index]])
            index[dense_index] = int(codebook_index_array[sparse_index])
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)
        index.reshape(shape)
        index_list.append(index)

    new_index_list, count_list, key_parameter = restructure_index(index_list, conv_layer_length, max_conv_bit,
                                                                  max_fc_bit)
    return new_index_list, count_list, key_parameter

def cluster_grad(count_list, net, index_list, max_conv_bit, max_fc_bit, conv_layer_length):
    params = list(net.parameters())
    # print('========Start========')
    for i in range(0, len(params), 2):
        # start = time.clock()
        param = params[i]
        grad_shape = param.grad.shape
        grad = param.grad
        grad = grad.view(-1)
        index = index_list[i]

        bias = params[i + 1]
        bias_grad_shape = bias.grad.shape
        bias_grad = bias.grad
        bias_grad = bias_grad.view(-1)
        bias_index = index_list[i + 1]

        half_index = int(i / 2)

        # Cluster grad using index, use mean of each class of grad to update weight
        cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
        for j in range(cluster_bits):
            sum_grad = grad[index[j]].sum()

            sum_grad += bias_grad[bias_index[j]].sum()
            count = count_list[half_index][j]
            mean_grad = 0
            if count != 0:
                mean_grad = sum_grad / count

            grad[index[j]] = mean_grad
            bias_grad[bias_index[j]] = mean_grad

        grad = grad.view(grad_shape)
        params[i].grad = grad.clone()

        bias_grad = bias_grad.view(bias_grad_shape)
        params[i + 1].grad = bias_grad.clone()



def update_codebook(net, codebook, conv_layer_length, max_conv_bit, max_fc_bit, key_parameter):
    params = list(net.parameters())
    for i in range(0, len(params), 2):
        # start = time.clock()
        param = params[i]
        param = param.view(-1)
        bias_param = params[i + 1]
        bias_param = bias_param.view(-1)
        layer = torch.cat((param, bias_param))
        half_index = int(i / 2)
        cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
        codebook_centroids = codebook.codebook_value[half_index]
        for j in range(cluster_bits):
            if key_parameter[half_index][j] is not None:
                tmp = key_parameter[half_index][j]
                codebook_centroids[j] = layer[tmp]


def train_codebook(key_parameter, count_list, use_cuda, max_conv_bit, max_fc_bit, conv_layer_length,
                   codebook, index_list, testloader, net, trainloader, criterion, optimizer,
                   epoch=1, epoch_step=25, ):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    for epoch in range(epoch):  # loop over the dataset multiple times
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

            cluster_grad(count_list, net, index_list, max_conv_bit, max_fc_bit, conv_layer_length)

            optimizer.step()  # update weight

            train_loss.append(loss.item())

            # TODO delete
            break

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        test(use_cuda, testloader, net)
        # print(list(net.parameters())[1])
        scheduler.step()
    update_codebook(net, codebook, conv_layer_length, max_conv_bit, max_fc_bit, key_parameter)


def save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, path, net):
    fc_merge_diff = []

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 405510 [0 0 0 0 0 0 0 0 0 0]

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
    # 5669 [2, 228, 211, 229, 76, 152, 23, 116, 111, 25]
    # 405510 [10, 11, 5, 6, 9, 7, 5, 7, 12, 5]
    # 544 [-0.11808116, -0.06328904, 0.1446653, 0.051914066, -0.03960273, -0.017428499, -0.017428499, 0.0050489083, 0.22879101, 0.051914066]

    length = len(fc_codebook_index)
    if length % 2 != 0:
        fc_codebook_index.append(0)

    fc_codebook_index = np.asarray(fc_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = []
    for i in range(math.floor((len(fc_codebook_index)) / 2)):
        fc_codebook_index_merge.append(
            (fc_codebook_index[2 * i] << 4) | fc_codebook_index[2 * i + 1])

    conv_codebook_index = np.asarray(conv_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = np.asarray(fc_codebook_index_merge, dtype=np.uint8)
    codebook_value = np.asarray(codebook_value, dtype=np.float32)

    # print(any(np.isnan(codebook_value)))

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index_merge), fc_codebook_index_merge[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 202755 [0 0 0 0 0 0 0 0 0 0]
    # 5669 [  2 228 211 229  76 152  23 116 111  25]
    # 202755 [200  66  71 152 140 171  86 151  87 197]
    # 544 [-0.11808116 -0.06328904  0.1446653   0.05191407 -0.03960273 -0.0174285
    #  -0.0174285   0.00504891  0.22879101  0.05191407]

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    codebook_value.dtype = np.uint8

    sparse_obj = np.concatenate((nz_num, conv_diff, fc_merge_diff, conv_codebook_index,
                                 fc_codebook_index_merge, codebook_value))
    sparse_obj.tofile(path)
