import torch
from torch import nn
import math
from dataclasses import dataclass, asdict
from typing import Literal, Optional


class OrthoTransform(nn.Module):
    def __init__(self, in_features, out_features, orthogonal_map: str = "householder", model: Optional[nn.Module]=None):
        super().__init__()
        self.ortho = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.orthogonal_(self.ortho.weight)
        self.transform = torch.nn.utils.parametrizations.orthogonal(
            self.ortho, orthogonal_map=orthogonal_map
        )


@dataclass
class OrthoConfig:
    orthogonal_map: str = "householder"
    """orthogonal map to use for the transform, can be 'householder', 'cayley', or 'matrix_exp'."""

    _target_: str = "reprpo.interventions.transforms.ortho.OrthoTransform"
