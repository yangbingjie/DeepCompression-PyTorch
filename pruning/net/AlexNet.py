import math
import torch.nn as nn
import torch.nn.functional as F
import pruning.function.Prune as prune


class AlexNet(prune.PruneModule):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = prune.MaskConv2Module(3, 96, 11, 4, 0)
        self.conv2 = prune.MaskConv2Module(96, 256, 5, 1, 2)
        self.conv3 = prune.MaskConv2Module(256, 384, 3, 1, 1)
        self.conv4 = prune.MaskConv2Module(384, 384, 3, 1, 1)
        self.conv5 = prune.MaskConv2Module(384, 256, 3, 1, 1)
        self.fc1 = prune.MaskLinearModule(6400, 4096)
        self.fc2 = prune.MaskLinearModule(4096, 4096)
        self.fc3 = prune.MaskLinearModule(4096, num_classes)
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
        x = nn.functional.dropout(x, p=self.drop_rate[0], training=True, inplace=False)
        x = F.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], training=True, inplace=False)
        x = self.fc3(x)
        return x

    def compute_dropout_rate(self):
        fc_list = [self.fc1, self.fc2, self.fc3]
        for index in range(0, 2):
            # Last Layer
            last_layer = fc_list[index]
            last_prune_num = 0
            last_total_num = 0
            if last_layer.bias is not None:
                bias_arr = (last_layer.bias_mask.data.cuda().numpy())
                last_prune_num = bias_arr.sum()
                last_total_num = bias_arr.size
            weight_arr = (last_layer.weight_mask.data.cuda().numpy())
            last_prune_num += weight_arr.sum()
            last_total_num += weight_arr.size

            # Next Layer
            next_layer = fc_list[index + 1]
            next_prune_num = 0
            next_total_num = 0
            if next_layer.bias is not None:
                bias_arr = (next_layer.bias_mask.data.cuda().numpy())
                next_prune_num = bias_arr.sum()
                next_total_num = bias_arr.size
            weight_arr = (next_layer.weight_mask.data.cuda().numpy())
            next_prune_num += weight_arr.sum()
            next_total_num += weight_arr.size

            #
            # p = 0.5 * math.sqrt(last_prune_num * next_prune_num / last_total_num * next_total_num)
            p = 0.5 * math.sqrt((last_prune_num / last_total_num) * (next_prune_num / next_total_num))
            print('The drop out rate is:', round(p,4))
            self.drop_rate[index] = p

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
