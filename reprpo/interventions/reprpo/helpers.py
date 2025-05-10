import torch
from einops import repeat

from torch import Tensor
from jaxtyping import Float

from baukit.nethook import get_module
import itertools
from reprpo.interventions.types import HS, Mask


def get_layer_paths(collection_keys, collection_layers):
    layer_paths = [
        [p.format(layer=layer) for p in collection_keys]
        for layer in collection_layers
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
    exp_weights = torch.exp(-cumsum_mask / 𝜏) * layer_attn_mask
    return exp_weights / exp_weights.sum(dim, keepdim=True)


def reduce_tokens_w_attention(
    x: HS, attn_mask: Mask, input_ids=None, tokenizer=None, 
    dim: int = 1, weight_tokens: bool = False, 
    filter_sinks: bool = True, sink_threshold: float = 1.0
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)
    with optional filtering of attention sinks"""
    
    layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    
    if filter_sinks:
        # Create enhanced mask that also filters attention sinks
        enhanced_mask = layer_attn_mask.clone()
        
        # # 1. Filter special tokens
        # for b in range(input_ids.shape[0]):  # For each item in batch
        #     for t in range(input_ids.shape[1]):  # For each token
        #         # FIXME use tokenizer dict get padding, unk, etc
        #         if input_ids[b, t] in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
        #             enhanced_mask[b, t] = 0
                    
        # 2. Filter high-magnitude tokens (potential attention sinks)
        with torch.no_grad():
            # Calculate token magnitudes
            token_magnitudes = torch.norm(x, dim=2)  # [b, t]
            
            # For each batch item, find tokens with abnormally high magnitudes
            for b in range(x.shape[0]):
                valid_tokens = (layer_attn_mask[b, :, 0] > 0)
                if valid_tokens.sum() > 0:
                    mean_mag = token_magnitudes[b, :].mean()
                    std_mag = token_magnitudes[b, :].std()
                    threshold = mean_mag + sink_threshold * std_mag
                    
                    # Mask out high-magnitude tokens
                    sink_tokens = token_magnitudes[b] > threshold
                    enhanced_mask[b, sink_tokens] = 0
        
        # Replace original mask with enhanced version
        layer_attn_mask = enhanced_mask.detach()
    
    if weight_tokens:
        # instead of taking the mean, let take an exponentially weighted mean
        exp_masked_weights = create_exp_weights(layer_attn_mask, dim)
        return (x * exp_masked_weights).sum(dim) / exp_masked_weights.sum(dim)
    else:
        return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)
