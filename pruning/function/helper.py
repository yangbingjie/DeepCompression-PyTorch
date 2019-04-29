import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import util.log as log
from scipy.sparse import csr_matrix
import torchvision.transforms as transforms


def load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers, name='MNIST', net_name='LeNet',
                 data_dir='./data'):
    train_set = None
    test_set = None
    transform_train = None
    transform_test = None
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3079,))
        ])
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True,
                                               download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False,
                                              download=True, transform=transform)
    elif name == 'CIFAR10':
        if net_name == 'VGG16':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2467, 0.2431, 0.2611)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2467, 0.2431, 0.2611)),
            ])
        elif net_name == 'AlexNet':
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2406, 0.2371, 0.2555)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2406, 0.2371, 0.2555)),
            ])
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                 download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                download=True, transform=transform_test)
    elif name == 'CIFAR100':
        if net_name == 'VGG16':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4381, 0.4178, 0.3773), (0.3002, 0.2870, 0.2933)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4381, 0.4178, 0.3773), (0.3002, 0.2870, 0.2933)),
            ])
        elif net_name == 'AlexNet':
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4381, 0.4178, 0.3773), (0.3002, 0.2870, 0.2933)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4381, 0.4178, 0.3773), (0.3002, 0.2870, 0.2933)),
            ])
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                  download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                 download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, **kwargs)
    return trainloader, testloader


def top_k_accuracy(outputs, labels, topk=(1,)):
    maxk = max(topk)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t().type_as(labels)
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def test(use_cuda, testloader, net, top_5=False):
    correct_1 = 0
    correct_5 = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            corr = top_k_accuracy(outputs, labels, topk=(1, 5))
            total += labels.size(0)
            correct_1 += corr[0]
            correct_5 += corr[1]
    top_1_accuracy = (100 * correct_1 / total)
    top_5_accuracy = (100 * correct_5 / total)
    if top_5:
        print('%.2f' % top_1_accuracy, '%.2f' % top_5_accuracy)
    else:
        print('%.2f' % top_1_accuracy)
        # print('Accuracy of the network on the test images: %.2f %%' % accuracy)
    return top_1_accuracy


def train(testloader, net, trainloader, criterion, optimizer, train_path, scheduler, max_accuracy, unit='K',
          save_sparse=False, epoch=1, use_cuda=True, auto_save=True, top_5=False):
    have_save = False
    for epoch in range(epoch):  # loop over the dataset multiple times
        # adjust_learning_rate(optimizer, epoch)
        train_loss = []
        # valid_loss = []
        net.train()
        # for inputs, labels in tqdm(trainloader):
        i = 0
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

        with torch.no_grad():
            # mean_train_loss = np.mean(train_loss)
            # print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
            # "Valid Loss: %5f" % mean_valid_loss
            accuracy = test(use_cuda, testloader, net, top_5)
            scheduler.step()
            if auto_save and accuracy > max_accuracy:
                if save_sparse:
                    save_sparse_model(net, train_path, unit)
                else:
                    torch.save(net.state_dict(), train_path)
                max_accuracy = accuracy
                have_save = True
    return have_save


# We store the index difference instead of the absolute position
# When we need an index difference larger than the bound, we padding filler zero to prevent overflow
def filler_zero(value, index, max_bits):
    last_index = -1
    max_bits_minus = max_bits - 1
    i = 0
    if index.size == 0:
        return index, index
    # Save filler zero num
    filler_num_array = []
    # Save filler zero position index
    filler_index_array = []
    while i < len(index):
        diff = index[i] - last_index - 1
        if diff > max_bits_minus:
            filler_num = math.floor(diff / max_bits)
            filler_num_array.append(filler_num)
            filler_index_array.append(i)
            last_index += filler_num * max_bits
        else:
            last_index = index[i]
            index[i] = diff
            i += 1

    new_len = value.size + sum(filler_num_array)
    new_value = np.empty(new_len, dtype=np.float32)
    new_index = np.empty(new_len, dtype=np.uint16)
    # index of new_index and new_value
    k = 0
    # index of filler_index_array and filler_num_array
    j = 0
    # index of index and value
    n = 0
    while k < new_len:
        if j < len(filler_index_array) and filler_index_array[j] == n:
            filler_num = filler_num_array[j]
            for m in range(filler_num):
                new_index[k] = max_bits_minus
                new_value[k] = 0
                k += 1
            j += 1
        else:
            new_index[k] = index[n]
            new_value[k] = value[n]
            n += 1
            k += 1

    return new_value, new_index


def save_sparse_model(net, path, unit):
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
    if unit == 'K':
        temp = 1024
    else:
        temp = 1048576
    print('The parameters are', round(nz_num.sum() / temp, 2), unit, layer_nz_num)
    conv_diff_array = np.asarray(conv_diff_array, dtype=np.uint8)
    fc_merge_diff = np.asarray(fc_merge_diff, dtype=np.uint8)
    conv_value_array = np.asarray(conv_value_array, dtype=np.float32)
    fc_value_array = np.asarray(fc_value_array, dtype=np.float32)

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    conv_value_array.dtype = np.uint8
    fc_value_array.dtype = np.uint8

    sparse_obj = np.concatenate((nz_num, conv_diff_array, fc_merge_diff, conv_value_array, fc_value_array))
    sparse_obj.tofile(path)
    log.log_file_size(path, unit)
