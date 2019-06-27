import torch
from torch import nn


class Squeeze_Layer(nn.Module):
    def __init__(self):
        super(Squeeze_Layer, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class Flatten_Layer(nn.Module):
    def __init__(self):
        super(Flatten_Layer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs
