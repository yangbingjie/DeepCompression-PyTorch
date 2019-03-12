import math
import torch
import torch.nn as nn
import torch.nn.functional as Function


class MaskLinearFunction(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinearFunction, self).__init__()
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

