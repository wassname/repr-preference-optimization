import torch
from dataclasses import dataclass, asdict
from .helpers import TransformByLayer

class NoneTransforms(TransformByLayer):
    Transform = torch.nn.Identity

@dataclass
class NoneConfig:
    pass

    def c(self, *args, **kwargs):
        return NoneTransforms(
            *args,
            **kwargs,
            **asdict(self),
        )
