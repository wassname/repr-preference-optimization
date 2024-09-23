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
        [p.format(layer=layer) for p in collection_keys]
        for layer in collection_layers_side
    ]
    layer_paths = list(itertools.chain(*layer_paths))
    return layer_paths


def validate_layer_paths(model, layer_paths):
    for p in layer_paths:
        get_module(model, p)


def detach_hsd(hs):
    """detach dict of hidden states"""
    return {k: v.detach() for k, v in hs.items()}

def create_exp_weights(layer_attn_mask, dim, 𝜏=50) -> torch.Tensor:
    """Create normalized exponentially decaying weights for a given sequence length."""
    cumsum_mask = layer_attn_mask.cumsum(dim)
    exp_weights = torch.exp(-cumsum_mask/𝜏) * layer_attn_mask
    return exp_weights / exp_weights.sum(dim, keepdim=True)

def reduce_tokens_w_attention(
    x: HS, attn_mask: Mask, dim: int = 1, weight_tokens: bool = False
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    if weight_tokens:
        # instead of taking the mean, let take an exponentially weighted mean

        # Create normalized exponentially decaying weights
        exp_masked_weights = create_exp_weights(layer_attn_mask, dim)

        return (x * exp_masked_weights).sum(dim) / exp_masked_weights.sum(dim)
    else:
        return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)
