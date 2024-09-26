import torch
from dataclasses import dataclass, asdict


@dataclass
class NoneConfig:
    pass

    def c(self, *args, **kwargs):
        return torch.nn.Identity(
            *args,
            **kwargs,
            **asdict(self),
        )
