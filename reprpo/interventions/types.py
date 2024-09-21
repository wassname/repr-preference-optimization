
import enum
from typing import Union, Dict, Callable
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass, asdict

HS = Float[Tensor, "b t h"]
Mask = Int[Tensor, "b t"]

@dataclass
class ReprPOModelOutput:
    hs: HS
    logits: Float[Tensor, "b l t h"]
    label_logprobs: Float[Tensor, "b l t"]
    mask: Mask

