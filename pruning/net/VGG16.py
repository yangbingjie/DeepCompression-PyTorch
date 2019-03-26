import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pruning.function.Prune as prune


class VGG16(prune.PruneModule):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()
        self.conv1 = prune.MaskConv2Module(3, 64, kernel_size=3, padding=1)
        self.conv2 = prune.MaskConv2Module(64, 64, kernel_size=3, padding=1)
        self.conv3 = prune.MaskConv2Module(64, 128, kernel_size=3, padding=1)
        self.conv4 = prune.MaskConv2Module(128, 128, kernel_size=3, padding=1)
        self.conv5 = prune.MaskConv2Module(128, 256, kernel_size=3, padding=1)
        self.conv6 = prune.MaskConv2Module(256, 256, kernel_size=3, padding=1)
        self.conv7 = prune.MaskConv2Module(256, 256, kernel_size=3, padding=1)
        self.conv8 = prune.MaskConv2Module(256, 512, kernel_size=3, padding=1)
        self.conv9 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.conv10 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.conv11 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.conv12 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.conv13 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.fc1 = prune.MaskLinearModule(512 * 7 * 7, 4096)
        self.fc2 = prune.MaskLinearModule(4096, 4096)
        self.fc3 = prune.MaskLinearModule(4096, num_classes)
        self.drop_rate = [0.5, 0.5]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop_rate[0], training=True, inplace=False)
        x = F.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop_rate[1], training=True, inplace=False)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, prune.MaskConv2Module):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def compute_dropout_rate(self):
        fc_list = [self.fc1, self.fc2, self.fc3]
        for index in range(0, 2):
            # Last Layer
            last_layer = fc_list[index]
            last_not_prune_num = 0
            last_total_num = 0
            if last_layer.bias is not None:
                bias_arr = last_layer.bias_mask.data
                last_not_prune_num = int(torch.sum(bias_arr))
                last_total_num = int(torch.numel(bias_arr))
            weight_arr = last_layer.weight_mask.data
            last_not_prune_num += int(torch.sum(weight_arr))
            last_total_num += int(torch.numel(weight_arr))

            # Next Layer
            next_layer = fc_list[index + 1]
            next_not_prune_num = 0
            next_total_num = 0
            if next_layer.bias is not None:
                bias_arr = next_layer.bias_mask.data
                next_not_prune_num = int(torch.sum(bias_arr.sum()))
                next_total_num = int(torch.numel(bias_arr))
            weight_arr = next_layer.weight_mask.data
            next_not_prune_num += int(torch.sum(weight_arr))
            next_total_num += int(torch.numel(weight_arr))

            # p = 0.5 * math.sqrt(last_not_prune_num * next_not_prune_num / last_total_num * next_total_num)
            p = 0.5 * math.sqrt((last_not_prune_num / last_total_num) * (next_not_prune_num / next_total_num))
            print('The drop out rate is:', round(p, 5))
            self.drop_rate[index] = p

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# net = VGG16()
# x = Variable(torch.FloatTensor(16, 3, 40, 40))
# y = net(x)
# print(y.data.shape)
# torch.Size([16, 1000])
