from reprpo.interventions.config import ExperimentConfig
from reprpo.interventions.losses import Losses, LossesType
from reprpo.interventions.transforms import Transforms, TransformType
from dataclasses import dataclass, field
from reprpo.interventions.reprpo.model import PL_REPRPO_MODEL
from typing import Optional, Tuple


@dataclass
class ReprPOConfig(ExperimentConfig):
    lr: float = 7e-5

    collection_layers: Optional[str] = 'range(.3,-1)'
    """layers to collect activations
    
    see `parse_collection_layers` which supports various formats:
    - "-2,-1" A comma-separated string like  to collect the last two layers
    - "range(3,10,2)" A string representing a range, e.g.,  which collects layers 3, 5, 7, 9
    - "range(0.5, 0.9, 2)" A shorthand range with fractions, e.g.,  which converts 0.5 to the 50% layer and 0.9 to the 90% layer
    - "0.3, 0.6, -1" A list of integers or floats
    """

    collection_keys_in: tuple = (
        ".*o_proj$",
        ".*out_proj$",
        ".*down_proj$",
    )
    """keys to collect inputs from uses regexp e.g. '.*o_proj$'"""

    collection_keys_out: tuple = (
        ".*q_proj$",
        ".*k_proj$",
        ".*v_proj$",
        ".*qkv_proj$",
        ".*gate_proj$",
        ".*up_proj$",
    )
    """keys to collect outputs from regexp."""

    collect_input: bool = True
    """use collection_keys_in? else use collection_keys_out."""

    collect_hs: bool = False
    """collect hidden states instead of activations"""

    loss: LossesType = field(default_factory=lambda: Losses.prefvec.value())
    """loss function"""

    transform: TransformType = field(default_factory=lambda: Transforms.ether.value())
    """transform function"""

    _cls = PL_REPRPO_MODEL

    _model_keys = [
        "lr",
        "collection_layers",
        "collection_keys_in",
        "collection_keys_out",
        "collect_input",
        "collect_hs",
        "loss",
        "transform",
    ]

    @property
    def _name(self):
        transform = type(self.transform).__name__.replace("Config", "")
        loss = type(self.loss).__name__.replace("Config", "").replace("Loss", "")
        h = "hs" if self.collect_hs else "side"
        return f"{h}-{transform}-{loss}"
