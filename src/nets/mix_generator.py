import torch
import torch.nn as nn

from . import utils
from .res_net_generator import ResNetBlock

class MixGenerator(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        
        self._first = utils.conv_block_7_1(3, 64, activation='relu', reflect=True)
        
        self._down = nn.ModuleList([
            utils.conv_block_4_2(64, 128, activation='leaky', normalize=False), # 128 -> 64
            utils.conv_block_4_2(128, 256, activation='leaky')])                # 64 -> 32
        
        self._neck = nn.Sequential(
            ResNetBlock(256, dropout=dropout),
            ResNetBlock(256, dropout=dropout),
            ResNetBlock(256),
            ResNetBlock(256))
        
        self._up = nn.ModuleList([
            utils.conv_transpose_block_4_2(512, 128, activation='relu'),        # 16 -> 32
            utils.conv_transpose_block_4_2(256, 64, activation='relu')])        # 32 -> 64
        
        self._last = utils.conv_block_7_1(64, 3, activation='tanh', normalize=False, reflect=True)
        
        self._optimizer = torch.optim.Adam(self.parameters(), utils.lr, utils.betas)
    
    def optimizer(self):
        return self._optimizer
    
    def forward(self, x):
        x = self._first(x)
        
        skips = []
        for downsample in self._down:
            x = downsample(x)
            skips.append(x)
        
        x = self._neck(x)
        
        for upsample, skip in zip(self._up, skips[::-1]):
            x = torch.cat((x, skip), 1)
            x = upsample(x)            
        
        return self._last(x)