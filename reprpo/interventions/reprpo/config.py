from reprpo.interventions.config import ExperimentConfig
from reprpo.interventions.losses import Losses, mse, LossesType
from reprpo.interventions.transforms import Transforms, TransformType
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from .model import PL_REPRPO_MODEL


@dataclass
class ReprPOConfig(ExperimentConfig):

    lr: float = 1e-4

    collection_layers_side: tuple=(10, 12, 14, 16, 18) 
    """layers to collect activations from in side layers."""

    # collection_layers_hs: tuple=(10, 20, 30)
    # """The layers to collect the hidden states from. Thisis for methods that operate on the redundant residual stream so only needs a couple of points of collection"""
    
    collection_keys_in: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    )
    """keys to collect inputs from."""

    collection_keys_out: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.q_proj",
        "base_model.model.model.layers.{layer}.self_attn.k_proj",
        "base_model.model.model.layers.{layer}.self_attn.v_proj",
        "base_model.model.model.layers.{layer}.mlp.gate_proj",
        "base_model.model.model.layers.{layer}.mlp.up_proj",
    )
    """keys to collect outputs from."""

    collect_input: bool = True
    """use collection_keys_in? else use collection_keys_out."""

    loss_fn: LossesType = mse
    """loss function"""

    transform: TransformType = Transforms.Ether.value
    """transform function"""

    _cls = PL_REPRPO_MODEL

    _model_keys = ['lr', 'collection_layers_side', 
    # 'collection_layers_hs',
                    'collection_keys_in', 'collection_keys_out', 'collect_input', 'loss_fn', 'transform', ]
