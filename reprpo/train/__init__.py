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
from .reprpo_side_ether import SideinETHER, SideoutETHER
from .reprpo_ether import ETHER
from .reprpo_hs_kl import HsKL
from .reprpo_hs_dist import HSDist
from .reprpo_side_dist import SideDist

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
    reprpo_side_ether = SideinETHER
    reprpo_sideout_ether = SideoutETHER
    reprpo_ether = ETHER
    reprpo_hra_kl = HsKL
    reprpo_hsdist = HSDist
    reprpo_hsdist_side = SideDist

MethodsUnion = Union[tuple(e.value for e in Methods)]
