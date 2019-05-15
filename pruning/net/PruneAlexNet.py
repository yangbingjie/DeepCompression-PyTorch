import torch.nn as nn
import torch.nn.functional as F
import pruning.function.prune as prune


class PruneAlexNet(prune.PruneModule):
    def __init__(self, num_classes=1000):
        super(PruneAlexNet, self).__init__()
        self.conv1 = prune.MaskConv2Module(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = prune.MaskConv2Module(64, 192, kernel_size=5, padding=2)
        self.conv3 = prune.MaskConv2Module(192, 384, kernel_size=3, padding=1)
        self.conv4 = prune.MaskConv2Module(384, 256, kernel_size=3, padding=1)
        self.conv5 = prune.MaskConv2Module(256, 256, kernel_size=3, padding=1)
        self.fc1 = prune.MaskLinearModule(9216, 4096)
        self.fc2 = prune.MaskLinearModule(4096, 4096)
        self.fc3 = prune.MaskLinearModule(4096, num_classes)
        self.drop_rate = [0.5, 0.5]
        self.fc_list = [self.conv5, self.fc1, self.fc2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(x.size(0), 9216)
        x = nn.functional.dropout(x, p=self.drop_rate[0], training=self.training, inplace=False)
        x = F.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], training=self.training, inplace=False)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
