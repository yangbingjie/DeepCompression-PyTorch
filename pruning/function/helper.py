import torch
import numpy as np
from pruning.function.csr import WeightCSR
from torch.autograd import Variable
import time


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

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def train(testloader, net, trainloader, valid_loader, criterion, optimizer, epoch=1, loss_accept=1e-3):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=1, verbose=True)
    for epoch in range(epoch):  # loop over the dataset multiple times
        train_loss = []
        valid_loss = []
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
        net.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                output = net(inputs)
                loss = criterion(output, labels)
                valid_loss.append(loss.item())

            mean_train_loss = np.mean(train_loss)
            mean_valid_loss = np.mean(valid_loss)
            print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss,
                  "Valid Loss: %5f" % mean_valid_loss)
            test(testloader, net)
            scheduler.step(mean_valid_loss)
            if mean_valid_loss < loss_accept:
                break


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
        if key.startswith('conv'):
            # 8 bits for conv layer index diff

            print('=======', key, 'start =========')
            start = time.clock()
            csr_matrix = WeightCSR(tensor, index_bits=8)
            diff_list, value_list = csr_matrix.tensor_to_csr()
            elapsed = (time.clock() - start)
            print('csr', round(elapsed, 5))

            start = time.clock()
            conv_diff_array.extend(diff_list)
            conv_value_array.extend(value_list)
            elapsed = (time.clock() - start)
            print('extend', round(elapsed, 5))
            print('=======', key, len(diff_list), 'end =======')

        else:
            # 4 bits for fc layer index diff
            print('=======', key, 'start =========')
            start = time.clock()
            csr_matrix = WeightCSR(tensor, index_bits=4)
            diff_list, value_list = csr_matrix.tensor_to_csr()
            elapsed = (time.clock() - start)
            print('csr', round(elapsed, 5))

            start = time.clock()
            fc_diff_array.extend(diff_list)
            fc_value_array.extend(value_list)
            elapsed = (time.clock() - start)
            print('extend', round(elapsed, 5))
            print('=======', key, len(diff_list), 'end =======')

        nz_num.append(csr_matrix.nz_num)

    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)
    fc_merge_diff = []
    for i in range(int(len(fc_diff_array) / 2) - 1):
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
    fc_diff_num = int(sum(nz_num[conv_layer_num:]) / 2)
    conv_diff = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_diff_num)
    value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num))

    print(fc_merge_diff[0:10])

    # Split 8 bits index to 4 bits index
    fc_diff = []
    for i in range(len(fc_merge_diff)):
        fc_diff.append(fc_merge_diff[i])  # first 4 bits
        fc_diff.append(fc_merge_diff[i])  # last 4 bits

    print(len(nz_num), nz_num)
    # 8 [  414    16 17721    30 58442    80   928     2]

    print(conv_diff.size, conv_diff[0:20])
    # 18181 [0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 1]

    print(fc_merge_diff.size, fc_diff[0:20])
    # 29726 [30, 30, 152, 152, 52, 52, 40, 40, 15, 15, 32, 32, 33, 33, 15, 15, 74, 74, 23, 23]

    print(value_array.size, value_array[0:20])
    # 77632 [-5.8470100e-31 -5.8545675e-31 -5.4793001e+19 -3.5765236e-35
    #  -7.3473500e+27 -2.2856749e-33 -2.6150384e+13 -2.1931591e+20
    #  -6.8449156e+18 -2.4266166e+04 -2.0393425e+11  4.6884696e-30
    #  -1.8285324e-32 -9.8099042e-24  1.1702544e-30  1.6096663e-19
    #  -7.4775139e-29 -7.4215263e-01 -5.6639610e-06 -4.3209863e-11]

    return conv_layer_num, nz_num, conv_diff, fc_diff, value_array
