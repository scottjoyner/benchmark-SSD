import torch
from torch import nn
import torch.nn.functional as F
from ssd.compat import compile_if_supported


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @compile_if_supported
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
