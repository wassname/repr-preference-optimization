from reprpo.interventions.config import ExperimentConfig
from reprpo.interventions.losses import Losses, LossesType
from reprpo.interventions.transforms import Transforms, TransformType
from dataclasses import dataclass, field
from reprpo.interventions.reprpo.model import PL_REPRPO_MODEL
from typing import Optional, Tuple


@dataclass
class ReprPOConfig(ExperimentConfig):
    lr: float = 7e-5

    collection_layers: Optional[str] = 'range(.3,-2)'
    """layers to collect activations from (none is parsed by `get_default_layers` which defaults to 33% onwards
    
    see `parse_collection_layers` which supports various formats:
    - A comma-separated string like "-2,-1" to collect the last two layers
    - A string representing a range, e.g., "range(3,10,2)"
    - A shorthand range with percentages, e.g., "0.5, 0.9, 2" which converts 0.5 to the 50% layer and 0.9 to the 90% layer
    - A list of integers or floats
    """

    # TODO change to regexp like peft
    collection_keys_in: tuple = (
        ".*o_proj$",
        ".*down_proj$",
    )
    """keys to collect inputs from"""

    # TODO change to regexp like peft
    collection_keys_out: tuple = (
        ".*q_proj$",
        ".*k_proj$",
        ".*v_proj$",
        ".*gate_proj$",
        ".*up_proj$",
    )
    """keys to collect outputs from."""

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
