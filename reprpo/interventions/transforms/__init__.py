import enum
from typing import Union

from .ortho import OrthoConfig
from .ether import ETHERConfig
from .hra import HRAConfig
from .none import NoneConfig
from .svd import SVDConfig


class Transforms(enum.Enum):
    ether = ETHERConfig
    ortho = OrthoConfig
    hra = HRAConfig
    none = NoneConfig
    svd = SVDConfig


TransformType = Union[tuple(e.value for e in Transforms)]
