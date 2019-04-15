import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler


def load_dataset(use_cuda, train_batch_size, test_batch_size, name='MNIST', data_dir='./data'):
    trainloader = None
    testloader = None
    kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}

    if name == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                             download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 **kwargs)
    elif name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 shuffle=False, **kwargs)

    return trainloader, testloader


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

def save_sparse_model(net, path):
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
            mat = csr_matrix(tensor.cpu().reshape(-1))
            bits = 8
            max_bits = 2 ** bits
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits)

            conv_diff_array.extend(diff_list)
            conv_value_array.extend(value_list)

        else:
            # 4 bits for fc layer index diff
            mat = csr_matrix(tensor.cpu().reshape(-1))
            bits = 4
            max_bits = 2 ** bits
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits)

            fc_diff_array.extend(diff_list)
            fc_value_array.extend(value_list)

        nz_num.append(len(diff_list))

    length = len(fc_diff_array)
    if length % 2 != 0:
        fc_diff_array.append(0)

    fc_diff_array = np.asarray(fc_diff_array, dtype=np.uint8)
    fc_merge_diff = []
    for i in range(int((len(fc_diff_array)) / 2)):
        fc_merge_diff.append((fc_diff_array[2 * i] << 4) | fc_diff_array[2 * i + 1])

    nz_num = np.asarray(nz_num, dtype=np.uint32)
    layer_nz_num = nz_num[0::2] + nz_num[1::2]
    print('The parameters are', round(nz_num.sum() / 1024, 2), 'K', layer_nz_num)
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
