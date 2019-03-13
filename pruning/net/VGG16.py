import math
import torch.nn as nn
import torch.nn.functional as F
from pruning.function.Prune import MaskLinearModule, PruneModule


class VGG16(PruneModule):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        x = nn.functional.dropout(x, p=self.drop_rate[0], inplace=False)
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
