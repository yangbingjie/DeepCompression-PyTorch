import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pruning.function.Prune as prune


class PruneAlexNet(prune.PruneModule, prune.DropoutNet):
    def __init__(self, num_classes=1000):
        super(PruneAlexNet, self).__init__()
        self.conv1 = prune.MaskConv2Module(3, 96, 11, 4, 0)
        self.conv2 = prune.MaskConv2Module(96, 256, 5, 1, 2)
        self.conv3 = prune.MaskConv2Module(256, 384, 3, 1, 1)
        self.conv4 = prune.MaskConv2Module(384, 384, 3, 1, 1)
        self.conv5 = prune.MaskConv2Module(384, 256, 3, 1, 1)
        self.fc1 = prune.MaskLinearModule(6400, 4096)
        self.fc2 = prune.MaskLinearModule(4096, 4096)
        self.fc3 = prune.MaskLinearModule(4096, num_classes)
        self.drop_rate = [0.5, 0.5]
        self.fc_list = [self.fc1, self.fc2, self.fc3]

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
        x = nn.functional.dropout(x, p=self.drop_rate[0], training=True, inplace=False)
        x = F.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], training=True, inplace=False)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
