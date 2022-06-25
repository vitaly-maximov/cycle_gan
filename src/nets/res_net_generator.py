import torch
import torch.nn as nn

from . import utils

# https://arxiv.org/pdf/1512.03385.pdf
class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=False):
        super().__init__()
        
        self._f = nn.Sequential(
            utils.conv_block_3_1(dim, dim, activation='relu', dropout=dropout, reflect=True),
            utils.conv_block_3_1(dim, dim, reflect=True))
    
    def forward(self, x):
        return x.add(self._f(x))

# https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
class ResNetGenerator(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        
        self._f = nn.Sequential(
            utils.conv_block_7_1(3, 64, activation='relu', reflect=True),
            utils.conv_block_4_2(64, 128, activation='relu'),
            utils.conv_block_4_2(128, 256, activation='relu'),
            ResNetBlock(256, dropout=dropout),
            ResNetBlock(256, dropout=dropout),
            ResNetBlock(256, dropout=dropout),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            utils.conv_transpose_block_4_2(256, 128, activation='relu'),
            utils.conv_transpose_block_4_2(128, 64, activation='relu'),
            utils.conv_block_7_1(64, 3, activation='tanh', normalize=False, reflect=True))
    
        self.apply(utils.init_weights)
        self._optimizer = torch.optim.Adam(self.parameters(), utils.lr, utils.betas)
    
    def optimizer(self):
        return self._optimizer
    
    def forward(self, x):
        return self._f(x)