import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate


class ShadowEstimation(nn.Module):
    
    def __init__(self, tau=.1, layers=8):
        super().__init__()
        self.register_buffer('tau', torch.nn.Parameter(torch.ones(1, 1, 1, 1) * tau))
        self.tau = torch.nn.Parameter(torch.ones(1, 1, 1, 1) * tau)
        self.layers = layers
        
    def parallel_sliding(self, dp, slope, dxdy):
        for t in range(self.layers): # 8 = ceil(lg(sqrt(2) * 128))
            dp = torch.max(dp, translate(
                dp - slope.view(-1, 1, 1, 1) * 2**t,
                -dxdy * 2**t,
                mode='bilinear',
                padding_mode='zeros'
            ))
        return dp
        
    def forward(self, l_dir, depth): # l_dir: (-1, 3), depth: (-1, 1, H, W), range: the width dimension is 1
        EPS = 1e-10
        depth = (depth - depth.min()) * depth.shape[-1] # depth is now >= 0, rescale to width dimension
        
        dxdy = F.normalize(l_dir[:,:2], dim=1)
        
        slope = -l_dir[:,2] * torch.rsqrt(1 - l_dir[:,2]**2 + EPS)
        max_along_trajectory = self.parallel_sliding(
            depth, slope, dxdy
        )
        # return torch.heaviside((depth - max_along_trajectory) , torch.ones(1, device=depth.device))
        return torch.exp((depth - max_along_trajectory) / (depth.shape[-1] * torch.abs(self.tau)))



