import enum
from typing import Union, Dict, Callable
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass, asdict

HS2 = Float[Tensor, "b h"]
HS = Float[Tensor, "b t h"]
Mask = Int[Tensor, "b t"]


@dataclass
class ReprPOModelOutput:
    hs: Dict[str, HS]
    logits: Float[Tensor, "b t v"]
    label_logprobs: Float[Tensor, "b"]
    mask: Mask
