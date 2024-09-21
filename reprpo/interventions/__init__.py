import enum
from typing import Union
from .dpo import DPOConfig
from .reprpo.config import ReprPOConfig


class Interventions(enum.Enum):
    dpo = DPOConfig
    reprpo = ReprPOConfig

InterventionType = Union[tuple(e.value for e in Interventions)]
