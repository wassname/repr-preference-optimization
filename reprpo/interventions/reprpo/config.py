from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from omegaconf import MISSING, OmegaConf

# defaults = [
#     "_self_",
#     {"loss": "prefvec"},
#     {"loss": "mse"},
#     {"loss": "rank"},
#      {"transform": "ether"},
#      {"transform": "none"},
    # {"intervention/reprpo/loss":"prefvec"},
# ]

@dataclass
class ReprPOConfig:
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_ : str = 'reprpo.interventions.reprpo.model.PL_REPRPO_MODEL'

    lr: float = 1e-4

    collection_layers_side: List[int] = field(default_factory=lambda:[10, 12, 14, 16, 18])
    """layers to collect activations from in side layers."""

    collection_keys_in: List[str] = field(default_factory=lambda:[
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    ])
    """keys to collect inputs from. if empty get hs."""

    collection_keys_out: List[str] = field(default_factory=lambda:[
        "base_model.model.model.layers.{layer}.self_attn.q_proj",
        "base_model.model.model.layers.{layer}.self_attn.k_proj",
        "base_model.model.model.layers.{layer}.self_attn.v_proj",
        "base_model.model.model.layers.{layer}.mlp.gate_proj",
        "base_model.model.model.layers.{layer}.mlp.up_proj",
    ])
    """keys to collect outputs from."""

    collect_input: bool = True
    """use collection_keys_in? else use collection_keys_out."""


    loss: Any = MISSING
    """loss function"""
    # loss: Losses = Losses.mse

    transform: Any = MISSING
    """transform function"""
    # transform: Transforms = Transforms.ether
