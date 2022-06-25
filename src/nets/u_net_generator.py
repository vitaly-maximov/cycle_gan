import torch
import torch.nn as nn

from . import utils

class UNetGenerator(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        
        self._down = nn.ModuleList([
            utils.conv_block_4_2(3, 64, activation='leaky', normalize=False),             # 128 -> 64
            utils.conv_block_4_2(64, 128, activation='leaky'),                            # 64 -> 32
            utils.conv_block_4_2(128, 256, activation='leaky'),                           # 32 -> 16
            utils.conv_block_4_2(256, 256, activation='leaky'),                           # 16 -> 8
            utils.conv_block_4_2(256, 256, activation='leaky')])                          # 8 -> 4
        
        self._up = nn.ModuleList([
            utils.conv_transpose_block_4_2(256, 256, activation='relu', dropout=dropout), # 4 -> 8
            utils.conv_transpose_block_4_2(512, 256, activation='relu', dropout=dropout), # 8 -> 16
            utils.conv_transpose_block_4_2(512, 128, activation='relu'),                  # 16 -> 32
            utils.conv_transpose_block_4_2(256, 64, activation='relu')])                  # 32 -> 64
        
        self._last = utils.conv_transpose_block_4_2(128, 3, activation="tanh", normalize=False)
        
        self.apply(utils.init_weights)
        self._optimizer = torch.optim.Adam(self.parameters(), utils.lr, utils.betas)
    
    def optimizer(self):
        return self._optimizer
    
    def forward(self, x):
        skips = []
        for downsample in self._down:
            x = downsample(x)
            skips.append(x)
        
        for upsample, skip in zip(self._up, skips[-2::-1]):
            x = upsample(x)
            x = torch.cat((x, skip), 1)
        
        return self._last(x)