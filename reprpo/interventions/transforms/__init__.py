import enum
from typing import Union

from .ortho import OrthoConfig
from .ether import EtherConfig
from .hra import HRAConfig

class Transforms(enum.Enum):
    Ortho = OrthoConfig
    Ether = EtherConfig
    HRA = HRAConfig

LossesType= Union[tuple(e.value for e in Transforms)]
