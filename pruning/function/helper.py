import torch
import numpy as np
import pickle
from pruning.function.csr import WeightCSR


def test(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # # TODO delete it
            break
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def train(net, trainloader, criterion, optimizer, epoch=1):
    for epoch in range(epoch):  # loop over the dataset multiple times
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
            if i % 2000 == 1999:  # print every mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            # # TODO delete it
            # if i == 100:
            #     break


def save_sparse_model(net, path):
    nz_num = []
    conv_diff_array = []
    fc_diff_array = []
    value_array = []
    for key, tensor in net.state_dict().items():
        if key.endswith('mask'):
            continue
        if key.startswith('conv'):
            # 8 bits for conv layer index diff
            csr_matrix = WeightCSR(tensor, index_bits=8)
            diff_list, value_list = csr_matrix.tensor_to_csr()
            conv_diff_array.extend(diff_list)
        else:
            # 4 bits for fc layer index diff
            csr_matrix = WeightCSR(tensor, index_bits=4)
            diff_list, value_list = csr_matrix.tensor_to_csr()
            fc_diff_array.extend(diff_list)
        nz_num.append(csr_matrix.nz_num)
        value_array.extend(value_list)
    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)
    fc_diff = []
    for i in range(int(len(fc_diff_array) / 2)):
        fc_diff.append(int(str(fc_diff_array[2 * i - 1]) + str(fc_diff_array[2 * i])))
    save_obj = {
        'nz_num': np.asarray(nz_num, dtype=np.uint32),
        'conv_diff': np.asarray(conv_diff_array, dtype=np.uint8),
        'fc_diff': np.asarray(fc_diff,dtype=np.uint8),
        'value': np.asarray(value_array, dtype=np.float32)
    }
    torch.save(save_obj, path)

# def load_sparse_model(net, path):
#     layers = filter(lambda x: 'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())  # 重构每一层
#     nz_num = np.fromfile(path, dtype=np.uint32, count=len(layers))
