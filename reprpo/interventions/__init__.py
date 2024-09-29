import enum
from typing import Union
from .dpo import DPOConfig
from .reprpo.config import ReprPOConfig
from .dpo_projgrad import ProjGradConfig
from .dpo_projbp import ProjBPConfig


class Interventions(enum.Enum):
    dpo = DPOConfig
    reprpo = ReprPOConfig
    projgrad = ProjGradConfig
    projbp = ProjBPConfig


InterventionType = Union[tuple(e.value for e in Interventions)]
