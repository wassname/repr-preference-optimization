import enum
from typing import Union

from .ortho import OrthoConfig
from .ether import ETHERConfig
from .hra import HRAConfig
from .none import NoneConfig
from .svd import SVDConfig
from .supressed import SupressedHSConfig


class Transforms(enum.Enum):
    ether = ETHERConfig
    # ortho = OrthoConfig # not compatible with bfloat16
    supr = SupressedHSConfig
    none = NoneConfig
    hra = HRAConfig
    svd = SVDConfig


TransformType = Union[tuple(e.value for e in Transforms)]
