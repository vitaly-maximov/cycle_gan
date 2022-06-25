import torch
import torch.nn as nn

from . import utils

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._f = nn.Sequential(
            utils.conv_block_4_2(3, 64, activation='leaky', normalize=False), # 128 -> 64
            utils.conv_block_4_2(64, 128, activation='leaky'),                # 64 -> 32
            utils.conv_block_4_2(128, 256, activation='leaky'),               # 32 -> 16
            utils.conv_block_3_1(256, 512, activation='leaky'),
            utils.conv_block_3_1(512, 1, normalize=False),
            nn.Flatten(),
            nn.Sigmoid())
        
        self.apply(utils.init_weights)
        self._optimizer = torch.optim.Adam(self.parameters(), utils.lr, utils.betas)
    
    def optimizer(self):
        return self._optimizer
    
    def forward(self, x):
        return self._f(x)