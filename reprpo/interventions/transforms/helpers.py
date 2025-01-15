from torch import nn
from typing import Dict
import torch
from jaxtyping import Float, Int
from torch import Tensor
from ..types import HS


class TransformByLayer(nn.Module):
    """Transforms dict of hidden states."""
    Transform: nn.Module

    def __init__(self, dim_sizes: Dict[str, int], model: nn.Module, **kwargs):
        super().__init__()
        self.transforms = torch.nn.ParameterDict(
            {
                k: self.Transform(dim_hs, dim_hs, model=model, **kwargs)
                for k, dim_hs in dim_sizes.items()
            }
        )

    def forward(self, x: Dict[str, HS]) -> Dict[str, HS]:
        for k, transform in self.transforms.items():
            x[k] = transform(x[k])
        return x
