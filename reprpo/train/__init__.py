import enum
from typing import Union
from .dpo import DPO
from .reprpo_svd import SVD
from .reprpo_hs import HS
from .reprpo_side import Sidein
from .reprpo_side import Sideout
from .reprpo_side_hra import SideinHRA, SideoutHRA
from .reprpo_ortho import Ortho
from .reprpo_hra import HRA

class Methods(enum.Enum):
    dpo = DPO
    reprpo_svd = SVD
    reprpo_hs = HS
    reprpo_side = Sidein
    reprpo_sideout = Sideout
    reprpo_side_hra = SideinHRA
    reprpo_sideout_hra = SideoutHRA
    reprpo_ortho = Ortho
    reprpo_hrank = HRA

MethodsUnion = Union[tuple(e.value for e in Methods)]
