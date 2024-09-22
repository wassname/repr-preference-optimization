from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal, Callable

import torch
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class NoneConfig:
    pass

    def c(self, *args, **kwargs):
        return torch.nn.Identity(
            *args,
            **kwargs,
            **asdict(self),
        )
