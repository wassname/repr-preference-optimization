import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat, reduce
import math
import warnings

from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from ..layers.hra import HRATransform

def collect_hs(hs):
    """The residual stream or hs of the diff of the hs."""
    hs = rearrange(list(hs), "l b t h -> l b t h")
    return rearrange(hs, "l b t h -> b l t h")

def reprpo_forward(model, input_ids, attn_mask, collection_layers):
    outs = model(
        input_ids,
        attention_mask=attn_mask,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    hs = collect_hs(outs.hidden_states)[:, collection_layers]

    logprobs = compute_logprobs(
        logits=outs.logits, labels=input_ids, selection_mask=attn_mask
    )

    return SimpleNamespace(hs=hs, logprobs=logprobs, logits=outs.logits)

# def mult_with_attention(
#     x: Float[Tensor, "b l t h"], attn_mask: Float[Tensor, "b t"], dim: int = 2
# ) -> Float[Tensor, "b l t h"]:
#     """x, weighted by the attention mask, over dim (token or batch)"""
#     layer_attn_mask = repeat(attn_mask, "b t -> b l t h", l=x.shape[1], h=1).detach()
#     return (x * layer_attn_mask) / layer_attn_mask.sum(dim, keepdim=True)

def mean_with_attention(
    x: Float[Tensor, "b l t h"], attn_mask: Float[Tensor, "b t"], dim: int = 2
) -> Float[Tensor, "b l h"]:
    """x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b l t h", l=x.shape[1], h=1).detach()
    return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)

def norm_mean(a, dim=-1, ord=2):
    # so normal torch norm is $||a||_2 = \sqrt{\sum_i a_i^2}$
    # we    want to do $||a||_2 = \sqrt{\mean_i a_i^2}$
    # we don't want to sum over hs, because layers have varying hs sizes
    # return torch.pow(torch.abs(a), ord).mean(dim)#.pow(1./ord)
    return torch.norm(a, p=ord, dim=dim)

def dist_ratio(a, b, attn, a_ref, b_ref, attn_ref, eps=1e-16, alpha = 1) -> Float[Tensor, "b"]:
    """Compute the distance ratio between two sets of hidden states, weighted by attention masks.
    
    log_dist_ratio = log(||a-b|| / ||a_ref - b_ref||)
    log_dist_ratio = log(||a-b||) - log(||a_ref - b_ref||))
    """


    dist = mean_with_attention(a-b, attn)  # reduces over tokens
    dist = norm_mean(dist, dim=-1)
    dist = dist + eps
    log_dist_ratio = torch.log(dist)

    # if provided with reference points, return the distance as a ratio to the reference distance
    if (a_ref is not None) and (b_ref is not None) and (attn_ref is not None):
        dist_ref = mean_with_attention(a_ref-b_ref, attn_ref).detach()
        dist_ref = norm_mean(dist_ref, dim=-1)
        dist_ref = dist_ref + eps # mean of 1e-5, very small
        log_dist_ratio -= torch.log(dist_ref).detach()
    
    assert torch.isfinite(log_dist_ratio).all()    
    return log_dist_ratio.nanmean(-1) * alpha


def compute_reprpo_hra_loss_batch(batch, model, alpha, collection_layers, transform, rel_loss=True):

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = reprpo_forward(
                model=model,
                input_ids=batch["chosen"],
                attn_mask=batch["chosen_mask"],
                collection_layers=collection_layers,
            )
            ref_rej = reprpo_forward(
                model=model,
                input_ids=batch["rejected"],
                attn_mask=batch["rejected_mask"],
                collection_layers=collection_layers,
            )

    model.train()
    pi_cho = reprpo_forward(
        model=model,
        input_ids=batch["chosen"],
        attn_mask=batch["chosen_mask"],
        collection_layers=collection_layers,
    )
    pi_rej = reprpo_forward(
        model=model,
        input_ids=batch["rejected"],
        attn_mask=batch["rejected_mask"],
        collection_layers=collection_layers,
    )
    assert torch.isfinite(pi_rej.hs).all()
    cho_attn_mask = batch["chosen_mask"]
    rej_attn_mask = batch["rejected_mask"]
    comb_attn_mask = cho_attn_mask * rej_attn_mask

    def res_det(hs):
        """use learnable transformation to get the residual part of the hs"""
        return transform(hs)

    # loss_retain: the representation of policy chosen responses should be closer to the reference chosen responses
    # and again we scale it using the reference model as a stable target
    # so should start at a -ve and go to 0 (as we optimize rr, not this)
    loss_retain = dist_ratio(
        ref_cho.hs.detach(),
        pi_cho.hs,
        comb_attn_mask,
        ref_cho.hs if rel_loss else None,
        ref_rej.hs,
        comb_attn_mask,
    ) 


    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    # start at 0 and go to -ve inf
    # start at log(1)=0 go to log(0)=-inf
    loss_reroute = dist_ratio(
        res_det(ref_cho.hs).detach(),
        res_det(pi_rej.hs),
        comb_attn_mask,
        # TODO what if we apply the transformation to the reference hs?
        # If we don't, we incentivize the model to learn a transformation with a small magnitude, and one where the distance can be reduced
        # if we do, we incentivize the model to learn an ortho transform that can be reduced
        res_det(ref_cho.hs) if rel_loss else None,
        res_det(ref_rej.hs),
        comb_attn_mask,
    )

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"])
    nll_loss_ratio = nll_loss / ref_nll_loss
    
    loss = (loss_reroute + loss_retain * alpha).nanmean()

    # get the dpo metrics for comparison
    _, info = compute_dpo_loss(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )

    def cosine_on_keys(hs1, hs2):
        return F.cosine_similarity(hs1, hs2, dim=-1).nanmean()
    
    with torch.no_grad():
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        # Lets monitor the comparitive norms of the decomposed parts
        hs = norm_mean(ref_cho.hs)
        hs_t = norm_mean(res_det(ref_cho.hs))
        hs_resid = norm_mean(ref_cho.hs - res_det(ref_cho.hs))
        info['hs_t/hs'] = (hs_t / hs).mean()
        info['hs_resid/hs'] = (hs_resid / hs).mean()
        info['hs_t/hs_resid'] = (hs_t / hs_resid).mean()

        # also the norm of the weights of the transformation
        info['transform_norm'] = torch.concat([norm_mean(p) for p in transform.parameters()]).mean()


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean() * alpha,
            nll_loss=nll_loss,
            ref_nll_loss=ref_nll_loss,
            nll_loss_ratio=nll_loss_ratio,
            **info,
        )
    assert torch.isfinite(loss)
    return loss, info

class PL_REPRPO_HRA_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers, r, apply_GS, rel_loss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        dim_hs = self._model.config.hidden_size
        self.transform = HRATransform(dim_hs, dim_hs, r=r, apply_GS=apply_GS)

    def _loss_fn(self, batch, model):
        return compute_reprpo_hra_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.transform,
            rel_loss=self.hparams.rel_loss
        )


@dataclass
class HRA(TrainingArgumentswCollection):
    """
    Transform: HRA (Householder Reflection Adaptation) along which to reroute the hidden states associated with the rejected responses. See: https://github.com/DaShenZi721/HRA
    """

    alpha: int = 0.0001
    """balancing retrain and reroute losses"""

    collection_layers: tuple=(10, 20) 
    """The layers to collect the hidden states from. HRA operates on the residual stream so only needs a couple of points of collection"""

    r: int = 256
    """The rank of HRA across different layers. Can be large as there is only one HRA matrix."""

    # lr: float = 1e-3

    apply_GS: bool = True
    """Whether to apply Gram-Schmidt orthogonalization."""

    rel_loss: bool = True

    _reprpo_class = PL_REPRPO_HRA_MODEL
    _model_keys = ['alpha', 'collection_layers', 'r', 'apply_GS', 'rel_loss' ]
