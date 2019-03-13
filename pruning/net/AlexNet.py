import math
import torch
import torch.nn as nn
from pruning.function.Prune import MaskLinearModule, PruneModule


class AlexNet(PruneModule):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 50)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


    def compute_dropout_rate(self):
        fc_list = [self.fc1, self.fc2]
        for index in range(0, 2):
            layer = fc_list[index]
            prune_num = 0
            basic = 0
            if layer.bias is not None:
                bias_arr = (layer.bias_mask.data.cpu().numpy())
                prune_num = bias_arr.sum()
                basic = bias_arr.size
            weight_arr = (layer.weight_mask.data.cpu().numpy())
            prune_num = prune_num + weight_arr.sum()
            basic = basic + weight_arr.size
            p = 0.5 * math.sqrt(prune_num / basic)
            print('The drop out rate is:', p)
            self.drop_rate[index] = p

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
