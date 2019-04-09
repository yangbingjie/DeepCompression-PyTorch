import torch
import numpy as np
from torch.autograd import Variable
import time
import math
from scipy.sparse import csr_matrix
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


def test(use_cuda, testloader, net):
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

    accuracy = (100 * correct / total)
    print('Accuracy of the network on the test images: %.2f %%' % accuracy)
    return accuracy


#
# def adjust_learning_rate(optimizer, epoch_num):
#     if epoch_num < 81:
#         lr = 0.01
#     elif epoch_num < 121:
#         lr = 0.001
#     else:
#         lr = 0.0001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def train(testloader, net, trainloader, criterion, optimizer, train_path, save_sparse=False, epoch=1, use_cuda=True,
        epoch_step=25, auto_save=True):
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                                        patience=1, verbose=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    max_accuracy = 0

    for epoch in range(epoch):  # loop over the dataset multiple times
        # adjust_learning_rate(optimizer, epoch)
        train_loss = []
        # valid_loss = []
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
            optimizer.step()  # update weight

            train_loss.append(loss.item())
        with torch.no_grad():
            mean_train_loss = np.mean(train_loss)
            # mean_valid_loss = np.mean(valid_loss)
            print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
            # "Valid Loss: %5f" % mean_valid_loss
            accuracy = test(use_cuda, testloader, net)
            scheduler.step()
            if auto_save and accuracy > max_accuracy:
                if save_sparse:
                    save_sparse_model(net, train_path)
                else:
                    torch.save(net.state_dict(), train_path)
                max_accuracy = accuracy



def filler_zero(value, index, max_bits):
    last_index = -1
    i = 0
    if index.size == 0:
        return index, index
    while last_index < index[-1]:
        diff = index[i] - last_index - 1
        if diff > max_bits - 1:
            filer_num = math.floor(diff / max_bits)
            for j in range(filer_num):
                value = np.insert(value, i, 0)
                index = np.insert(index, i, max_bits - 1)
            last_index += filer_num * max_bits
            i += filer_num
        else:
            last_index = index[i]
            index[i] = diff
            i += 1
    return value, index


# # test filler zero
# a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               0, 0, 0, 0, 0, 0, 0, 0, 0, 3.4,
#               0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0,
#               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               0, 0, 0, 0, 0, 0, 1.7])
# tensor = torch.from_numpy(a)
# mat = csr_matrix(tensor)
# print(mat.data)
# print(mat.indices)
# value_list, diff_list = filler_zero(mat.data, mat.indices, 8)
# print(value_list)
# print(diff_list)
# diff_list = np.array(diff_list, dtype=np.uint8)
# print(diff_list)


def save_sparse_model(net, path):
    # torch.save(net.state_dict(), path)
    # for key, tensor in net.state_dict().items():
    #     if key.endswith('mask'):
    #         a = int(torch.sum(tensor))
    #         b = int(torch.numel(tensor))
    #         print(a, b, a / b)

    nz_num = []
    conv_diff_array = []
    fc_diff_array = []
    conv_value_array = []
    fc_value_array = []
    for key, tensor in net.state_dict().items():
        if key.endswith('mask'):
            continue
        # print('=======', key, 'start =========')
        if key.startswith('conv'):
            # 8 bits for conv layer index diff
            # start = time.clock()
            mat = csr_matrix(tensor.cpu().reshape(-1))
            bits = 8
            max_bits = 2 ** bits
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits)
            # elapsed = (time.clock() - start)
            # print('csr', round(elapsed, 5))

            # start = time.clock()
            conv_diff_array.extend(diff_list)
            conv_value_array.extend(value_list)
            # elapsed = (time.clock() - start)
            # print('extend', round(elapsed, 5))

        else:
            # 4 bits for fc layer index diff
            # start = time.clock()
            mat = csr_matrix(tensor.cpu().reshape(-1))
            bits = 4
            max_bits = 2 ** bits
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits)

            # print(sum(diff_list) + len(diff_list))

            # elapsed = (time.clock() - start)
            # print('csr', round(elapsed, 5))

            # start = time.clock()
            fc_diff_array.extend(diff_list)
            fc_value_array.extend(value_list)
            # elapsed = (time.clock() - start)
            # print('extend', round(elapsed, 5))

        # print('=======', key, len(diff_list), 'end =======')
        nz_num.append(len(diff_list))

    # print(nz_num)
    # print(len(conv_diff_array), conv_diff_array[-10:])
    # print(len(fc_diff_array), fc_diff_array[-10:])
    # print(len(conv_value_array), conv_value_array[-10:])
    # print(len(fc_value_array), fc_value_array[-10:])
    # [292, 17, 8213, 15, 77747, 65, 818, 1]
    # 8537 [3, 1, 2, 0, 3, 2, 0, 1, 2, 4]
    # 78631 [4, 2, 12, 1, 10, 0, 3, 0, 6, 8]
    # 8537 [0.055003658, -0.0518913, -0.057878394, 0.047473326, -0.07086759, -0.071428634, -0.060436048, -0.06711546, -0.0698091, -0.06924898]
    # 78631 [0.13233908, 0.16305041, -0.171971, -0.1353672, 0.16033891, -0.19598335, -0.114601016, -0.32042998, -0.12170218, 0.14367148]

    # layer_index = fc_diff_array[0:0 + nz_num[4]]
    # print(sum(layer_index) + len(layer_index))
    # print(sum(item > 15 for item in fc_diff_array))
    # print(sum(item < 0 for item in fc_diff_array))

    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)

    fc_diff_array = np.array(fc_diff_array, dtype=np.uint8)
    fc_merge_diff = []
    for i in range(int((len(fc_diff_array)) / 2)):
        fc_merge_diff.append((fc_diff_array[2 * i] << 4) | fc_diff_array[2 * i + 1])

    nz_num = np.asarray(nz_num, dtype=np.uint32)
    print('The parameters are', round(nz_num.sum() / 1024, 2), 'K')
    conv_diff_array = np.asarray(conv_diff_array, dtype=np.uint8)
    fc_diff = np.asarray(fc_merge_diff, dtype=np.uint8)
    conv_value_array = np.asarray(conv_value_array, dtype=np.float32)
    fc_value_array = np.asarray(fc_value_array, dtype=np.float32)

    # # TODO delete it
    # # -------------------------------------------------------
    # fc_diff1 = []
    # bits = 4
    # max_bits = 2 ** bits
    # for i in range(len(fc_diff)):
    #     fc_diff1.append(int(fc_diff[i] / max_bits))  # first 4 bits
    #     fc_diff1.append(fc_diff[i] % max_bits)  # last 4 bits
    # fc_num_sum = sum(nz_num[4:])
    # if fc_num_sum % 2 != 0:
    #     fc_diff1 = fc_diff1[:fc_num_sum]
    # fc_diff1 = np.asarray(fc_diff1, dtype=np.uint8)
    #
    # layer_index = fc_diff1[0:0 + nz_num[4]]
    # print(sum(layer_index) + len(layer_index))
    #
    # # 要么相同，要么fc_diff_array比fc_diff1多1
    # print(len(fc_diff_array), len(fc_diff1))
    # count = 0
    # for i in range(len(fc_diff1)):
    #     if fc_diff_array[i] != fc_diff1[i]:
    #         count += 1
    # print(count)
    # # -------------------------------------------------------
    #
    # print(nz_num)
    # print(conv_diff_array.size, conv_diff_array[-10:])
    # print(fc_diff.size, fc_diff[-10:])
    # print(conv_value_array.size, conv_value_array[-10:])
    # print(fc_value_array.size, fc_value_array[-10:])
    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 39316 [ 17 242  34  50 164  44  26   3   6 128]
    # 8537 [ 0.05500366 -0.0518913  -0.05787839  0.04747333 -0.07086759 -0.07142863
    #  -0.06043605 -0.06711546 -0.0698091  -0.06924898]
    # 78631 [ 0.13233908  0.16305041 -0.171971   -0.1353672   0.16033891 -0.19598335
    #  -0.11460102 -0.32042998 -0.12170218  0.14367148]

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    conv_value_array.dtype = np.uint8
    fc_value_array.dtype = np.uint8

    sparse_obj = np.concatenate((nz_num, conv_diff_array, fc_diff, conv_value_array, fc_value_array))
    sparse_obj.tofile(path)


