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
        self.ortho = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.orthogonal_(self.ortho.weight)
        self.transform = torch.nn.utils.parametrizations.orthogonal(
            self.ortho, orthogonal_map=orthogonal_map
        )


    
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
