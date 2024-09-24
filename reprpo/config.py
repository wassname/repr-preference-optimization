
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Any, Optional, List
from omegaconf import MISSING, OmegaConf
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.interventions.reprpo.config import ReprPOConfig
from reprpo.interventions.dpo import DPOConfig

# defaults = [
#     "_self_",
#     {"intervention": "reprpo"},

# ]

@dataclass
class ExperimentConfig:
    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genies_preferences"""

    # defaults: List[Any] = field(default_factory=lambda: defaults)
    intervention: Any = MISSING

    dataset: str = "us_history_textbook"
    """train dataset."""

    verbose: int = 0

    dev: bool = False
    """fast run"""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = False

    batch_size: int = 16

    n_samples: int = 1800 * 2
    eval_samples: Optional[int] = None
    max_length: int = 196
    max_prompt_length: int = 96
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="expconf", node=ExperimentConfig)
    cs.store(group="intervention", name="dpo", node=DPOConfig)
    cs.store(group="intervention", name="reprpo", node=ReprPOConfig)
    for k in Losses:
        cs.store(group="intervention.loss", name=k.name, node=k.value)
    for k in Transforms:
        cs.store(group="intervention.transform", name=k.name, node=k.value)
    return cs

# cs = get_config_store()
