from kan import *
import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, device):
        super(Embedder, self).__init__()
        self.embedder = KAN(width=[16, [16, 4], 16], grid=3, k=3, device=device)

    def forward(self, x):
        x = self.embedder(x)
        return x
