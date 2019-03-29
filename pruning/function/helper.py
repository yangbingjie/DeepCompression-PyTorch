import torch
import numpy as np
from torch.autograd import Variable
import time
from scipy.sparse import csr_matrix
import torch.optim.lr_scheduler as lr_scheduler


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
          accuracy_accept=99, epoch_step=25):
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                                        patience=1, verbose=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    max_accuracy = 0

    for epoch in range(epoch):  # loop over the dataset multiple times
        # adjust_learning_rate(optimizer, epoch)
        train_loss = []
        # valid_loss = []
        net.train()
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
            optimizer.step()  # update weight

            train_loss.append(loss.item())
        # net.eval()
        with torch.no_grad():
            # for inputs, labels in valid_loader:
            #     inputs = inputs.cuda()
            #     labels = labels.cuda()
            #     output = net(inputs)
            #     loss = criterion(output, labels)
            #     valid_loss.append(loss.item())

            mean_train_loss = np.mean(train_loss)
            # mean_valid_loss = np.mean(valid_loss)
            print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
            # "Valid Loss: %5f" % mean_valid_loss
            accuracy = test(testloader, net)
            scheduler.step()
            if accuracy > max_accuracy:
                if save_sparse:
                    save_sparse_model(net, train_path)
                else:
                    torch.save(net.state_dict(), train_path)
                max_accuracy = accuracy
            if accuracy >= accuracy_accept:
                break


def filler_zero(value, index, bits):
    max_bits = 2 ** bits
    last_index = -1
    i = 0
    while i < len(index):
        diff = index[i] - last_index
        if diff > max_bits:
            filer_num = int(diff / max_bits)
            for j in range(filer_num):
                value = np.insert(value, i, 0)
                index = np.insert(index, i, max_bits - 1)
            last_index += filer_num * max_bits
            i += filer_num
        else:
            last_index = index[i]
            index[i] = diff - 1
            i += 1
    return value, index


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
        print('=======', key, 'start =========')
        if key.startswith('conv'):
            # 8 bits for conv layer index diff
            start = time.clock()
            mat = csr_matrix(tensor.cpu().reshape(-1))
            value_list, diff_list = filler_zero(mat.data, mat.indices, 8)
            elapsed = (time.clock() - start)
            print('csr', round(elapsed, 5))

            start = time.clock()
            conv_diff_array.extend(diff_list)
            conv_value_array.extend(value_list)
            elapsed = (time.clock() - start)
            print('extend', round(elapsed, 5))

        else:
            # 4 bits for fc layer index diff
            start = time.clock()
            mat = csr_matrix(tensor.cpu().reshape(-1))
            value_list, diff_list = filler_zero(mat.data, mat.indices, 4)
            elapsed = (time.clock() - start)
            print('csr', round(elapsed, 5))

            start = time.clock()
            fc_diff_array.extend(diff_list)
            fc_value_array.extend(value_list)
            elapsed = (time.clock() - start)
            print('extend', round(elapsed, 5))

        print('=======', key, len(diff_list), 'end =======')
        nz_num.append(len(diff_list))

    # print(nz_num)
    # print(len(conv_diff_array), conv_diff_array[0:20])
    # print(len(fc_diff_array), fc_diff_array[0:20])
    # print(len(conv_value_array), conv_value_array[0:20])
    # print(len(fc_value_array), fc_value_array[0:20])

    # [368, 20, 14706, 28, 195060, 223, 1986, 5]
    # 15122 [1, 0, 0, 3, 1, 0, 0, 0, 0, 1, 1, 2, 0, 4, 0, 1, 0, 0, 0, 0]
    # 197274 [0, 3, 1, 0, 1, 0, 1, 0, 0, 0, 4, 2, 0, 0, 0, 1, 0, 4, 0, 2]
    # 15122 [-0.058692038, -0.20448187, -0.06760995, 0.07252452, 0.08391143, 0.25181824, 0.3066377, 0.15797374, 0.08447786, -0.28649807, 0.16029164, -0.14801468, -0.23152983, 0.25150803, 0.10125634, -0.061441414, 0.24004352, -0.21027368, -0.5110624, -0.3133713]
    # 197274 [-0.021946901, 0.020857502, 0.020772176, 0.028257154, 0.031773686, 0.036305174, -0.027708048, -0.030588016, 0.034516267, 0.024990581, -0.02094119, 0.019869655, 0.023522332, -0.024851361, -0.022399157, -0.026098453, -0.02566959, 0.03573141, -0.029567914, -0.021915225]

    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)
    fc_merge_diff = []
    for i in range(int(len(fc_diff_array) / 2)):
        fc_merge_diff.append((fc_diff_array[2 * i] << 4) + fc_diff_array[2 * i + 1])
    nz_num = np.asarray(nz_num, dtype=np.uint32)
    conv_diff_array = np.asarray(conv_diff_array, dtype=np.uint8)
    fc_diff = np.asarray(fc_merge_diff, dtype=np.uint8)
    conv_value_array = np.asarray(conv_value_array, dtype=np.float32)
    fc_value_array = np.asarray(fc_value_array, dtype=np.float32)

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    conv_value_array.dtype = np.uint8
    fc_value_array.dtype = np.uint8

    # print(nz_num)
    # print(conv_diff_array.size, conv_diff_array[0:20])
    # print(fc_diff.size, fc_diff[0:20])
    # print(conv_value_array.size, conv_value_array[0:20])
    # print(fc_value_array.size, fc_value_array[0:20])

    # [112   1   0   0  20   0   0   0 114  57   0   0  28   0   0   0 244 249 2   0 223   0   0   0 194   7   0   0   5   0   0   0]
    # 15122 [1 0 0 3 1 0 0 0 0 1 1 2 0 4 0 1 0 0 0 0]
    # 98637 [ 3 16 16 16  0 66  0  1  4  2 20 34  0 66  0  0 67  0  2  0]
    # 60488 [ 16 103 112 189 178  99  81 190  22 119 138 189 188 135 148  61 193 217 171  61]
    # 789096 [253 201 179 188  90 221 170  60 105  42 170  60 140 123 231  60  32  37 2  61]

    sparse_obj = np.concatenate((nz_num, conv_diff_array, fc_diff, conv_value_array, fc_value_array))
    sparse_obj.tofile(path)


def load_sparse_model(net, path):
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
    fc_diff_num = int((sum(nz_num[conv_layer_num:]) + 1) / 2)
    conv_diff = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_diff_num)
    conv_value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num[:conv_layer_num]))
    fc_value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num[conv_layer_num:]))

    # print(fc_merge_diff.size, fc_merge_diff[0:10])
    # 98637 [ 3 16 16 16  0 66  0  1  4  2]

    # Split 8 bits index to 4 bits index
    fc_diff = []
    bits = 4
    max_bits = 2 ** bits
    for i in range(len(fc_merge_diff)):
        fc_diff.append(int(fc_merge_diff[i] / max_bits))  # first 4 bits
        fc_diff.append(fc_merge_diff[i] % max_bits)  # last 4 bits
    #
    # print(nz_num)
    # print(conv_diff.size, conv_diff[0:20])
    # print(len(fc_diff), fc_diff[0:20])
    # print(conv_value_array.size, conv_value_array[0:20])
    # print(fc_value_array.size, fc_value_array[0:20])

    # [   368     20  14706     28 195060    223   1986      5]
    # 15122 [1 0 0 3 1 0 0 0 0 1 1 2 0 4 0 1 0 0 0 0]
    # 197274 [0, 3, 1, 0, 1, 0, 1, 0, 0, 0, 4, 2, 0, 0, 0, 1, 0, 4, 0, 2]
    # 15122 [-0.05869204 -0.20448187 -0.06760995  0.07252452  0.08391143  0.25181824
    #   0.3066377   0.15797374  0.08447786 -0.28649807  0.16029164 -0.14801468
    #  -0.23152983  0.25150803  0.10125634 -0.06144141  0.24004352 -0.21027368
    #  -0.5110624  -0.3133713 ]
    # 197274 [-0.0219469   0.0208575   0.02077218  0.02825715  0.03177369  0.03630517
    #  -0.02770805 -0.03058802  0.03451627  0.02499058 -0.02094119  0.01986966
    #   0.02352233 -0.02485136 -0.02239916 -0.02609845 -0.02566959  0.03573141
    #  -0.02956791 -0.02191523]

    return conv_layer_num, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array
