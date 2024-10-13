import enum
from typing import Union
from .rank import RankLossConfig
from .prefvec import PrefVecLossConfig
from .mse import MSELossConfig


class Losses(enum.Enum):
    """
    Define losses that take in ReprPOModelOutput
    and output loss, info
    """

    prefvec = PrefVecLossConfig
    mse = MSELossConfig
    rank = RankLossConfig


LossesType = Union[tuple(e.value for e in Losses)]
