import torch
import torch.nn.functional as F
import torch.nn as nn


class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
    
if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    m = GeM()
    r = m(x)
    print(r.shape)