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
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = prune.MaskConv2Module(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = prune.MaskConv2Module(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = prune.MaskConv2Module(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = prune.MaskConv2Module(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = prune.MaskConv2Module(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = prune.MaskConv2Module(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = prune.MaskConv2Module(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = prune.MaskConv2Module(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.fc1 = prune.MaskLinearModule(512 * 7 * 7, 4096)
        self.fc2 = prune.MaskLinearModule(4096, 4096)
        self.fc3 = prune.MaskLinearModule(4096, num_classes)
        self.drop_rate = [0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[0], training=True, inplace=False)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[1], training=True, inplace=False)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[2], training=True, inplace=False)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[3], training=True, inplace=False)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[4], training=True, inplace=False)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[5], training=True, inplace=False)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[6], training=True, inplace=False)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.drop_rate[7], training=True, inplace=False)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.functional.dropout(x, p=self.drop_rate[8], training=True, inplace=False)
        x = self.fc2(x)
        x = F.relu(x)
        x = nn.functional.dropout(x, p=self.drop_rate[9], training=True, inplace=False)
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
        last_layer_array = [self.conv1, self.conv3, self.conv5, self.conv6, self.conv8, self.conv9, self.conv11,
                            self.conv12, self.fc1, self.fc2]
        next_layer_array = [self.conv2, self.conv4, self.conv6, self.conv7, self.conv9, self.conv10, self.conv12,
                            self.conv13, self.fc2, self.fc3]
        for index in range(len(self.drop_rate)):
            # Last Layer
            last_layer = last_layer_array[index]
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
            next_layer = next_layer_array[index]
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
# # torch.Size([16, 1000])
