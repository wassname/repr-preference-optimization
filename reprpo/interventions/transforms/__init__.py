import enum
from typing import Union

from .ortho import OrthoConfig
from .ether import ETHERConfig
from .hra import HRAConfig

class Transforms(enum.Enum):
    Ortho = OrthoConfig
    Ether = ETHERConfig
    HRA = HRAConfig

LossesType= Union[tuple(e.value for e in Transforms)]
