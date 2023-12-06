import torch.nn as nn
import torch
import math
# from set_device import device
import numpy as np
# device = 'cpu'
device = 'cuda:1'

class GaussianFourier(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale):
        super().__init__()
        self.scale = scale

        # torch[num_input_channels, mapping_size]
        self.B = torch.randn((num_input_channels, mapping_size)) * scale
        self.B = self.B.to(device)
    
    def forward(self, x):
        # calc 2 * pi * B * v
        # make [2, 160000]
        x = x.permute(1, 0)
        calc_result = self.B.T @ x
        calc_result *= 2 * math.pi
        # [16, 160000]
        return torch.cat([torch.cos(calc_result), torch.sin(calc_result)], dim = 0).permute(1, 0)

    def get_B(self, path):
        B_stored = self.B.cpu().detach().numpy()
        np.save(path, B_stored)

    def set_B(self, path):
        B_loaded = np.load(path)
        self.B = torch.tensor(B_loaded, device=device, dtype=torch.float32)

    











