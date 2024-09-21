import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from baukit.nethook import TraceDict, get_module
import itertools
from reprpo.interventions.types import HS, Mask

def get_layer_paths(collection_keys, collection_layers_side):
    layer_paths = [
        [p.format(layer=layer) for p in collection_keys] for layer in collection_layers_side
    ]
    layer_paths = list(itertools.chain(*layer_paths))
    return layer_paths


def validate_layer_paths(model, layer_paths):
    for p in layer_paths:
        get_module(model, p)


def detach_hsd(hs):
    """detach dict of hidden states"""
    return {k: v.detach() for k, v in hs.items()}


def mean_tokens_w_attention(
    x: HS, attn_mask: Mask, dim: int = 1
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)
