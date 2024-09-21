import enum
from typing import Union, Dict
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from ..types import ReprPOModelOutput
from .rank import RankLossConfig
from .prefvec import PrefVecLossConfig
from .mse import MSELossConfig


class Losses(enum.Enum):
    """
    Define losses that take in ReprPOModelOutput
    and output loss, info
    """
    rank = RankLossConfig
    prefvec = PrefVecLossConfig
    mse = MSELossConfig


LossesType= Union[tuple(e.value for e in Losses)]
