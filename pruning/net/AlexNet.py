import math
import torch.nn as nn
import torch.nn.functional as F
from pruning.function.Prune import MaskLinearModule, PruneModule


class AlexNet(PruneModule):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 0)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384,384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384,256, 3, 1, 1)
        self.fc1 = MaskLinearModule(9216, 4096),
        self.fc2 = nn.Linear(4096, 4096),
        self.fc3 = nn.Linear(4096, 50)
        self.drop_rate = [0.5, 0.5]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop_rate[0], inplace=False)
        x = F.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], inplace=False)
        x = self.fc3(x)
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
