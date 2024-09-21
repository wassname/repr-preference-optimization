
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.interventions.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.interventions.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict, get_module
from dataclasses import dataclass
import itertools
from reprpo.interventions.types import ReprPOModelOutput, HS, Mask



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


def reprpo_forward_baukit(model, input_ids, attn_mask, layer_paths, collect_input=True):

    reprs = {}
    with TraceDict(
        model,
        layer_paths,
        retain_input=collect_input,
        retain_output=(not collect_input),
        retain_grad=True,
    ) as ret:
        outs = model(
            input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        for p in layer_paths:
            if collect_input:
                reprs[p] = ret[p].input
            else:
                reprs[p] = ret[p].output
            assert torch.isfinite(reprs[p]).all()

    logprobs = compute_logprobs(
        logits=outs.logits, labels=input_ids, selection_mask=attn_mask
    )

    return ReprPOModelOutput(hs=reprs, logits=outs.logits, label_logprobs=logprobs, mask=attn_mask)

from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms

class PL_REPRPO_MODEL(PL_MODEL):
    def __init__(self, *args, collection_layers_side, collect_input, collection_keys_in: tuple=None, collection_keys_out: tuple=None,  
                 loss_config: Losses,
                 transform_config: Transforms,
                 **kwargs):
        super().__init__(*args, **kwargs)
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers_side)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        # we do one learnable householder roation per layer
        if collect_input:
            hra_sizes = {p:get_module(self._model, p).in_features for p in self.hparams.layer_paths}
        else:
            hra_sizes = {p:get_module(self._model, p).out_features for p in self.hparams.layer_paths}
        
        self.transforms = torch.nn.ParameterDict({
            k: transform_config.c(dim_hs, dim_hs) for k,dim_hs in hra_sizes.items()})
        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)

    def _loss_fn(self, batch, model):
        h = self.hparams
        model.eval()
        with torch.no_grad():
            with model.disable_adapter():
                ref_cho = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["chosen"],
                    attn_mask=batch["chosen_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                )
                ref_rej = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["rejected"],
                    attn_mask=batch["rejected_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                )

        model.train()
        pi_cho = reprpo_forward_baukit(
            model=model,
            input_ids=batch["chosen"],
            attn_mask=batch["chosen_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
        )
        pi_rej = reprpo_forward_baukit(
            model=model,
            input_ids=batch["rejected"],
            attn_mask=batch["rejected_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
        )
        
        return h.loss_config.c(
            pi_cho=pi_cho,
            pi_rej=pi_rej,
            ref_cho=ref_cho,
            ref_rej=ref_rej,
            batch=batch,
        )


@dataclass
class ReprPOConfig(ModelConfigBase):
    alpha: float = 0.1
    """balanced retain and reroute"""

    collection_layers_side: tuple=(10, 12, 14, 16, 18) 
    """layers to collect activations from in side layers."""

    collection_layers_hs: tuple=(10, 20, 30)
    """The layers to collect the hidden states from. Thisis for methods that operate on the redundant residual stream so only needs a couple of points of collection"""
    
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

    loss_fn: Losses
    """loss function"""

    transform: Optional[Transforms] = None
    """transform function"""

    _cls = PL_REPRPO_MODEL
