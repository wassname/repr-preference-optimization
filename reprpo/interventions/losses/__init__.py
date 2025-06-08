import enum
from typing import Union
from .rank import RankLossConfig
from .innerdpo import InnerDPOLossConfig
from .mse import MSELossConfig


class Losses(enum.Enum):
    """
    Define losses that take in ReprPOModelOutput
    and output loss, info
    """

    InnerDPO = InnerDPOLossConfig
    rank = RankLossConfig
    mse = MSELossConfig


LossesType = Union[tuple(e.value for e in Losses)]
