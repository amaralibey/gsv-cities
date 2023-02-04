import torch
import torch.nn.functional as F
import torch.nn as nn
from gem import GeM

class CosPlace(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor 
    """
    def __init__(self, in_dim, out_dim):
        super(CosPlace, self).__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    m = CosPlace(2048, 512)
    r = m(x)
    print(r.shape)