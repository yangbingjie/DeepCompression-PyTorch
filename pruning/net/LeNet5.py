import math
import torch.nn as nn
import torch.nn.functional as F
from pruning.function.MaskLinearFunction import MaskLinearModule

class LeNet5(MaskLinearModule):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = MaskLinearModule(4 * 4 * 50, 500)
        self.fc2 = MaskLinearModule(500, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nn.functional.dropout(x, p=0.5, training=True, inplace=False)
        x = F.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=True, inplace=False)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
