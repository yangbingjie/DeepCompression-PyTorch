import torch
import numpy as np
from pruning.function.csr import WeightCSR


def test(testloader, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    net = net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # # TODO delete
            # break
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))



def train(net, trainloader, criterion, optimizer, epoch=1, log_frequency=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lambda1 = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    net.to(device)
    for epoch in range(epoch):  # loop over the dataset multiple times
        scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # backward
            # 将已修剪的连接的梯度置为零
            for name, p in net.named_parameters():
                if name.endswith('mask') or p.grad is None or p.data is None:
                    continue
                # p.data数据可能在gpu里，.cpu()可以将值拷贝一份到cpu
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < 1e-6, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()  # update weight

            # print statistics
            running_loss += loss.item()
            if i % log_frequency == log_frequency - 1:  # print every mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_frequency))
                running_loss = 0.0
            # # TODO delete
            # break


def save_sparse_model(net, path):
    nz_num = []
    conv_diff_array = []
    fc_diff_array = []
    value_array = []
    total = 0
    for key, tensor in net.state_dict().items():
        if key.endswith('mask'):
            total += tensor.cpu().numpy().reshape(-1).size
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
    print('prune rate: ', round(total / sum(nz_num), 5))
    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)
    fc_diff = []
    for i in range(int(1 - len(fc_diff_array) / 2)):
        fc_diff.append(int(str(fc_diff_array[2 * i - 1]) + str(fc_diff_array[2 * i])))
    save_obj = {
        'nz_num': np.asarray(nz_num, dtype=np.uint32),
        'conv_diff': np.asarray(conv_diff_array, dtype=np.uint8),
        'fc_diff': np.asarray(fc_diff, dtype=np.uint8),
        'value': np.asarray(value_array, dtype=np.float32)
    }
    torch.save(save_obj, path)

