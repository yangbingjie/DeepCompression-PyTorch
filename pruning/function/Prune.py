import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Function
from torch.nn.modules.module import Module

class MaskLinearModule(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_mask = nn.Parameter(torch.ones((out_features, in_features)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias_mask = nn.Parameter(torch.ones(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            return Function.linear(input, self.weight * self.weight_mask, self.bias * self.bias_mask)
        else:
            return Function.linear(input, self.weight * self.weight_mask)

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

class PruneModule(Module):
    def prune_layer(self, sensitivity=0.5):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * sensitivity
                print('Pruning layer', name, ' threshold: ', round(threshold, 2))
                module.prune(threshold)
