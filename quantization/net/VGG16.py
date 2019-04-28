import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        kernel_size = 3
        padding = 1
        in_channels = 3
        names = self.__dict__['_modules']
        vgg16_cfg = cfg['D']
        i = 0
        for layer in vgg16_cfg:
            if layer != 'M':
                i += 1
                names['conv' + str(i)] = nn.Conv2d(in_channels, layer,
                                                   kernel_size=kernel_size, padding=padding)
                in_channels = layer
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.drop_rate = [0.5, 0.5]

    def forward(self, x):
        vgg16_cfg = cfg['D']
        names = self.__dict__['_modules']
        i = 0
        for layer in vgg16_cfg:
            if layer == 'M':
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                i += 1
                x = names['conv' + str(i)](x)
                x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop_rate[0], training=self.training, inplace=False)
        x = F.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], training=self.training, inplace=False)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# import torch
# from torch.autograd import Variable
# net = VGG16()
# x = Variable(torch.FloatTensor(16, 3, 40, 40))
# y = net(x)
# print(y.data.shape)
# torch.Size([16, 1000])
