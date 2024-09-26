from reprpo.interventions.config import ExperimentConfig
from reprpo.interventions.losses import Losses, mse, LossesType
from reprpo.interventions.transforms import Transforms, TransformType
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from .model import PL_REPRPO_MODEL


@dataclass
class ReprPOConfig(ExperimentConfig):
    lr: float = 4e-4

    collection_layers_side: tuple = (10, 12, 14, 16, 18)
    """layers to collect activations from in side layers."""

    collection_keys_in: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    )
    """keys to collect inputs from"""

    collection_keys_out: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.q_proj",
        "base_model.model.model.layers.{layer}.self_attn.k_proj",
        "base_model.model.model.layers.{layer}.self_attn.v_proj",
        "base_model.model.model.layers.{layer}.mlp.gate_proj",
        "base_model.model.model.layers.{layer}.mlp.up_proj",
    )
    """keys to collect outputs from."""

    collect_input: bool = False
    """use collection_keys_in? else use collection_keys_out."""

    collect_hs: bool = False
    """collect hidden states instead of activations"""

    loss: LossesType = Losses.prefvec.value
    """loss function"""

    transform: TransformType = Transforms.ether.value
    """transform function"""

    _cls = PL_REPRPO_MODEL

    _model_keys = [
        "lr",
        "collection_layers_side",
        "collection_keys_in",
        "collection_keys_out",
        "collect_input",
        "collect_hs",
        "loss",
        "transform",
    ]

    @property
    def _name(self):
        transform = type(self.transform).__name__.replace('Config', '')
        loss = type(self.loss).__name__.replace('Config', '').replace('Loss', '')
        h = 'side' if len(self.collection_layers_side) > 0 else 'hs'
        return f"{h}-{transform}-{loss}"
