import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

class MaskModule(Module):
    def prune(self, threshold):
        weight_dev = self.weight.device
        weight_mask_dev = self.weight_mask.device
        bias_dev = self.bias.device
        bias_mask_dev = self.bias_mask.device
        weight = self.weight.data.cpu().numpy()
        weight_mask = self.weight_mask.data.cpu().numpy()
        bias = self.bias.data.cpu().numpy()
        bias_mask = self.bias_mask.data.cpu().numpy()
        new_weight_mask = np.where(abs(weight) < threshold, 0, weight_mask)
        self.weight.data = torch.from_numpy(weight * new_weight_mask).to(weight_dev)
        self.weight_mask.data = torch.from_numpy(new_weight_mask).to(weight_mask_dev)
        if self.bias is not None:
            new_bias_mask = np.where(abs(bias) < threshold, 0, bias_mask)
            self.bias.data = torch.from_numpy(bias * new_bias_mask).to(bias_dev)
            self.bias_mask.data = torch.from_numpy(new_bias_mask).to(bias_mask_dev)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class MaskLinearModule(MaskModule):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True).cuda()
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False).cuda()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True).cuda()
            self.bias_mask = nn.Parameter(torch.ones(out_features).byte(), requires_grad=False).cuda()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        if self.bias is not None:
            return F.linear(input, self.weight * self.weight_mask.float(), self.bias * self.bias_mask.float())
        else:
            return F.linear(input, self.weight * self.weight_mask.float())


class MaskConv2Module(MaskModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *(kernel_size, kernel_size)), requires_grad=True).cuda()
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False).cuda()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True).cuda()
            self.bias_mask = nn.Parameter(torch.ones(out_channels).byte(), requires_grad=False).cuda()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        weight_dev = self.weight.device
        weight = self.weight.data.cpu().numpy()
        weight_mask = self.weight_mask.data.cpu().numpy()
        weight = torch.from_numpy(weight * weight_mask).to(weight_dev)
        input = input.cuda()
        if self.bias is not None:
            bias_dev = self.bias.device
            bias = self.bias.data.cpu().numpy()
            bias_mask = self.bias_mask.data.cpu().numpy()
            bias = torch.from_numpy(bias * bias_mask).to(bias_dev)
            return F.conv2d(input, weight, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(input, weight, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)


class PruneModule(Module):
    # prune_mode: prune 'conv' or prune 'fc'
    # 'not': is not retrain
    # 'conv': retrain conv layer, fix fc layer
    # 'fc': retrain fc layer, fix conv layer
    def prune_layer(self, sensitivity=None, prune_mode='not'):
        if prune_mode == 'not':
            return
        if sensitivity is None:
            sensitivity = {
                'fc': 0.9,
                'conv1': 0.5,
                'conv': 0.7,
            }
        print('===== prune', prune_mode, '=====')
        for name, module in self.named_modules():
            if name.startswith(prune_mode):
                # The pruning threshold is chosen as a quality parameter multiplied
                # by the standard deviation of a layer's weight
                if name == 'conv1':
                    s = sensitivity[name]
                elif name.startswith('fc'):
                    s = sensitivity['fc']
                else:
                    s = sensitivity['conv']
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print('Pruning layer', name, ' threshold: ', round(threshold, 4))
                module.prune(threshold)
        print('====== prune end ======')

    # fix_mode: fix 'conv' or 'fc'
    # 'not': is not retrain
    # 'conv': retrain conv layer, fix fc layer
    # 'fc': retrain fc layer, fix conv layer
    def fix_layer(self, fix_mode='not'):
        if fix_mode == 'not':
            return
        print('===== fix mode is', fix_mode, '=====')
        for name, p in self.named_parameters():
            if name.endswith('mask'):
                continue
            elif name.startswith(fix_mode):
                p.requires_grad = False
                # print('Fix', name)
            else:
                p.requires_grad = True
                # print('Open', name)
        # print('======''fix mode end', '=======')
