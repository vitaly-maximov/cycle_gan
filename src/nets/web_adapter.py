import torch
import torch.nn as nn

class WebAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
    
    def forward(self, x):
        x = ((x / 255) - 0.5) / 0.5                    # [0; 255] -> [-1; 1]
        x = x.permute(2, 0, 1)                         # [128, 128, 4] -> [4, 128, 128]
        x = torch.narrow(x, dim=0, start=0, length=3)  # [4, 128, 128] -> [3, 128, 128]
        x = x.reshape(1, 3, 128, 128)

        x = self._model(x)

        x = x.reshape(3, 128, 128)        
        x = torch.cat((x, torch.ones(1, 128, 128)), 0) # [3, 128, 128] -> [4, 128, 128]
        x = x.permute(1, 2, 0)                         # [4, 128, 128] -> [128, 128, 4]
        x = 255 * (0.5 * x + 0.5)                      # [-1; 1] -> [0; 255]

        return x