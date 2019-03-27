import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class MaskModule(Module):
    def prune(self, threshold):
        new_weight_mask = torch.where(abs(self.weight.data) < threshold,
                                      torch.zeros_like(self.weight.data).float().cuda(),
                                      self.weight_mask.float())
        self.weight.data = self.weight * new_weight_mask.float()
        self.weight_mask.data = new_weight_mask
        if self.bias is not None:
            new_bias_mask = torch.where(abs(self.bias.data) < threshold,
                                        torch.zeros_like(self.bias.data).byte().cuda(),
                                        self.bias_mask)
            self.bias.data = self.bias * new_bias_mask.float()
            self.bias_mask.data = new_bias_mask


class MaskLinearModule(MaskModule):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
            self.bias_mask = nn.Parameter(torch.ones(out_features).byte(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        weight = self.weight * self.weight_mask.float()
        if self.bias is not None:
            return F.linear(input, weight, self.bias * self.bias_mask.float())
        else:
            return F.linear(input, weight)
        nn.Linear

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


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
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *(kernel_size, kernel_size)),
                                   requires_grad=True)
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
            self.bias_mask = nn.Parameter(torch.ones(out_channels).byte(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        weight = self.weight * self.weight_mask.float()
        if self.bias is not None:
            bias = self.bias * self.bias_mask.float()
            return F.conv2d(input, weight, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(input, weight, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


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
                'fc': 0.77,
                'conv1': 0.3,
                'conv': 0.5,
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
                threshold = module.weight.data.std() * s
                # print('Pruning layer', name, ' threshold: ', round(threshold, 4))
                module.prune(threshold)
        # print('====== prune end ======')

    # fix_mode: fix 'conv' or 'fc'
    # 'not': is not retrain
    # 'conv': retrain conv layer, fix fc layer
    # 'fc': retrain fc layer, fix conv layer
    def fix_layer(self, net, fix_mode='not'):
        if fix_mode == 'not':
            return
        print('===== fix mode is', fix_mode, '=====')
        for name, p in net.named_parameters():
            if name.endswith('mask') or name.starswith('bn'):
                continue
            elif name.startswith(fix_mode):
                p.requires_grad = False
            else:
                p.requires_grad = True
        print('===== fix mode end', '======')
