from torch.nn import functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch
import numpy as np


class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, leaky = 0.2, dropout = 0.5):
        super(Discriminator, self).__init__()
        dim = input_dim # 수정
        self.packdim = dim
        seq = []

        for item in list(dis_dims):
            seq += [Linear(dim, item)]
            if leaky == 0:
                seq += [ReLU()]
            else:
                seq += [LeakyReLU(leaky)]
            
            if dropout != 0:
                seq += [Dropout(dropout)]
            
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % 1 == 0
        return self.seq(input.view(-1,  self.packdim)) # 수정
