"""

Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

See the following references for more information:
- https://arxiv.org/html/2406.19384v1
    - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4). 

- https://arxiv.org/pdf/2401.12181
    > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
"""

import torch
from torch import nn
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict
from .helpers import TransformByLayer
from ..types import HS

class SupressedHSTransform(nn.Module):
    def __init__(self, dim_sizes: Dict[str, int], model: nn.Module, **kwargs):
        super().__init__()

    def forward(self, x: Dict[str, HS]) -> Dict[str, HS]:
        keys = sorted([k for k in x.keys()])   
        hs = torch.stack([x[k] for k in keys], dim=1)
        hs = hs.diff(dim=1).clamp(min=None, max=0)
        return {k: hs[:, i] for i, k in enumerate(keys[1:])}

@dataclass
class SupressedHSConfig:
    """
    Get only the activations that are supressed between layers. These are the ones values that are not used for prediction but may be used internally.
    """

    def c(
        self,
        *args,
        **kwargs,
    ):
        return SupressedHSTransform(*args, **kwargs, **asdict(self))
