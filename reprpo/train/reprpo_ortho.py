import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat, reduce

from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArguments, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools





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

def mult_with_attention(
    x: Float[Tensor, "b l t h"], attn_mask: Float[Tensor, "b t"], dim: int = 2
) -> Float[Tensor, "b l t h"]:
    """x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b l t h", l=x.shape[1], h=1).detach()
    return (x * layer_attn_mask) / layer_attn_mask.sum(dim, keepdim=True)

def mean_with_attention(
    x: Float[Tensor, "b l t h"], attn_mask: Float[Tensor, "b t"], dim: int = 2
) -> Float[Tensor, "b l h"]:
    """x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b l t h", l=x.shape[1], h=1).detach()
    return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)

def norm(a, dim=-1):
    # return torch.abs(a).mean(dim)
    return torch.pow(a, 2).mean(dim)

def dist_ratio(a, b, attn, a_ref, b_ref, attn_ref, eps=1e-16) -> Float[Tensor, "b l"]:


    dist = mean_with_attention(a-b, attn)  # reduces over tokens
    # dist = reduce(dist, "b l t h -> b l h", torch.nanmean)
    # dist = torch.norm_except_dim(dist, pow=1, dim=-1)
    dist = norm(dist, dim=-1)
    # dist = torch.norm(dist, p=2, dim=-1)
    dist = dist + eps

    dist_ref = mean_with_attention(a_ref-b_ref, attn_ref).detach()
    # dist_ref = reduce(dist_ref, "b l t h -> b l h", torch.nanmean)
    # dist_ref = torch.norm_except_dim(dist_ref, pow=1, dim=-1) + eps
    dist_ref = norm(dist_ref, dim=-1)
    # dist_ref = torch.norm(dist_ref, p=2, dim=-1)
    assert torch.isfinite(dist_ref).all()
    dist_ref = dist_ref + eps # mean of 1e-5, very small

    # get the ratio in log space to avoid div by zero
    # NOTE: for retain dist start at zero
    # log_dist_ratio = dist / (dist_ref+ eps)
    log_dist_ratio = torch.log(dist) - torch.log(dist_ref).detach()
    assert torch.isfinite(log_dist_ratio).all()

    alpha = 1
    return log_dist_ratio.nanmean(-1) * alpha


def compute_reprpo_orth_loss_batch(batch, model, alpha, collection_layers, transform):

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
    # loss_retain: the representation of policy chosen responses should be closer to the reference chosen responses
    # and again we scale it using the reference model as a stable target
    # so should start at a -ve and go to 0 (as we optimize rr, not this)
    loss_retain = dist_ratio(
        ref_cho.hs.detach(),
        pi_cho.hs,
        comb_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        comb_attn_mask,
    ) 

    def res_det(hs):
        """use learnable transformation to get the residual part of the hs"""
        return transform(hs)

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    # start at 0 and go to -ve inf
    # start at log(1)=0 go to log(0)=-inf
    loss_reroute = dist_ratio(
        res_det(ref_cho.hs).detach(),
        res_det(pi_rej.hs),
        comb_attn_mask,
        (ref_cho.hs),
        (ref_rej.hs),
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
        hs = norm(ref_cho.hs)
        hs_t = norm(res_det(ref_cho.hs))
        hs_resid = norm(ref_cho.hs - res_det(ref_cho.hs))
        info['hs_t/hs'] = (hs_t / hs).mean()
        info['hs_resid/hs'] = (hs_resid / hs).mean()
        info['hs_t/hs_resid'] = (hs_t / hs_resid).mean()

        # also the norm of the weights of the transformation
        info['transform_norm'] = norm(transform.weight).mean()


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

class PL_REPRPO_ORTHO_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        dim_hs = self._model.config.hidden_size
        t = nn.Linear(dim_hs, dim_hs, bias=False)
        torch.nn.init.orthogonal_(t.weight)
        self.transform = torch.nn.utils.parametrizations.orthogonal(t, orthogonal_map="householder")

    def _loss_fn(self, batch, model):
        return compute_reprpo_orth_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.transform
        )


@dataclass(frozen=True)
class ReprPOOrthoTrainingArguments(TrainingArguments):
    """weights retrain and reroute losses"""
    alpha: int = 0.01

    collection_layers: tuple=(10, 20) 

    _reprpo_class = PL_REPRPO_ORTHO_MODEL
    _model_keys = ['alpha', 'collection_layers', ]
