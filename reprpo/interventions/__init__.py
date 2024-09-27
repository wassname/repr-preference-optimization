import enum
from typing import Union
from .dpo import DPOConfig
from .reprpo.config import ReprPOConfig
from .dpo_projgrad import DPOProjGradConfig


class Interventions(enum.Enum):
    dpo = DPOConfig
    reprpo = ReprPOConfig
    projgrad = DPOProjGradConfig


InterventionType = Union[tuple(e.value for e in Interventions)]
