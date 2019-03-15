import torch
import pickle
import numpy as np
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


def train(net, trainloader, criterion, optimizer):
    for epoch in range(1):  # loop over the dataset multiple times
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
            # if i == 500:
            #     break

# TODO
def save_sparse_model(net, path):
    for key, tensor in net.state_dict().items():
        # print(key, tensor.shape)
        # 8 bits for conv layer and 5 bits for fc layer
        csr_matrix = WeightCSR(tensor, index_bits=8 if key.startswith('conv') else 5)
        csr_matrix.tensor_to_csr()
        # TODO delete
        break



# def load_sparse_model(net, path):
#     layers = filter(lambda x: 'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())  # 重构每一层
#     nz_num = np.fromfile(path, dtype=np.uint32, count=len(layers))
