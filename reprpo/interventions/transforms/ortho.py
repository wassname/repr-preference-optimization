import torch
from torch import nn
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from .helpers import TransformByLayer

class OrthoTransform(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        orthogonal_map: str = "householder",
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        ortho = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.orthogonal_(ortho.weight)
        self.transform = torch.nn.utils.parametrizations.orthogonal(
            ortho, orthogonal_map=orthogonal_map
        )

    def forward(self, x):
        return self.transform(x)


    
class OrthoTransforms(TransformByLayer):
    Transform = OrthoTransform

@dataclass
class OrthoConfig:
    orthogonal_map: Literal["householder", "cayley", "matrix_exp"] = "householder"

    def c(
        self,
        *args,
        **kwargs,
    ):
        return OrthoTransforms(*args, **kwargs, **asdict(self))
