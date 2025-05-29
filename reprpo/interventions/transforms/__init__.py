import enum
from typing import Union

from .ether import ETHERConfig
from .none import NoneConfig
from .supressed import SupressedHSConfig


class Transforms(enum.Enum):
    ether = ETHERConfig
    supr = SupressedHSConfig
    none = NoneConfig


TransformType = Union[tuple(e.value for e in Transforms)]
