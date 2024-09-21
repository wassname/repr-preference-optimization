import torch
from torch import nn
import math
from dataclasses import dataclass, asdict
from typing import Literal

class OrthoTransform(nn.Module):
    def __init__(self, in_features, out_features, orthogonal_map:str="householder"):
        super().__init__()
        self.ortho = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.orthogonal_(self.ortho.weight)
        self.transform = torch.nn.utils.parametrizations.orthogonal(self.ortho, orthogonal_map=orthogonal_map)

@dataclass(frozen=True)
class OrthoConfig:
    orthogonal_map: Literal["householder", "cayley", "matrix_exp"] = "householder"

    @property
    def c(self):
        return OrthoTransform(**asdict(self))
