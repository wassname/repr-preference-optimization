import enum
from typing import Union
from .rank import RankLossConfig
from .innerdpo import InnerDPOLossConfig
from .mse import MSELossConfig
from .topk import TopKLossConfig
from .topkl import TopKLLossConfig


class Losses(enum.Enum):
    """
    Define losses that take in ReprPOModelOutput
    and output loss, info
    """

    InnerDPO = InnerDPOLossConfig
    rank = RankLossConfig
    mse = MSELossConfig
    topk = TopKLossConfig
    topkl = TopKLLossConfig


LossesType = Union[tuple(e.value for e in Losses)]
