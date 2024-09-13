import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat, reduce
import math
import warnings

from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArguments, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools


class HRA(nn.Module):
    """
    see
    - https://github.com/huggingface/peft/blob/54be5a3db61748d698ca2e6b55bcfef229a9b475/src/peft/tuners/hra/layer.py#L197
    """

    def __init__(self, in_features, out_features, 
                 r=8, apply_GS=False):
        super(HRA, self).__init__()
        

        self.hra_r = r
        self.apply_GS = apply_GS
        self.hra_u = nn.Parameter(torch.randn(in_features, r))

        self.reset_hra_parameters()

    def reset_hra_parameters(self):
        if self.hra_r % 2 != 0:
            warnings.warn("The symmetric initialization can NOT be performed when r is odd!")
            nn.init.kaiming_uniform_(self.hra_u, a=math.sqrt(5))
        else:
            shape = self.hra_u.shape
            half_u = torch.zeros(shape[0], shape[1] // 2)
            nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
            self.hra_u = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))

    def get_delta_weight(self, reverse: bool = False) -> torch.Tensor:
        rank = self.hra_r
        apply_GS = self.apply_GS
        opt_u = self.hra_u
        shape = opt_u.shape

        if apply_GS:
            weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
            for i in range(1, rank):
                ui = opt_u[:, i].view(-1, 1)
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()

        else:
            opt_u = opt_u / opt_u.norm(dim=0)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
            if reverse:
                indices = range(rank - 1, -1, -1)
            else:
                indices = range(rank)

            for i in indices:
                ui = opt_u[:, i].view(-1, 1)
                weight = weight @ (torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * ui @ ui.t())

        return weight
    
    def forward(self, input):
        delta_weight = self.get_delta_weight()
        return torch.matmul(input, delta_weight)



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


def compute_reprpo_hra_loss_batch(batch, model, alpha, collection_layers, transform):

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
        info['transform_norm'] = torch.concat([norm(p) for p in transform.parameters()]).mean()


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
    def __init__(self, *args, alpha, collection_layers, r, apply_GS, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        dim_hs = self._model.config.hidden_size
        self.transform = HRA(dim_hs, dim_hs, r=r, apply_GS=apply_GS)

    def _loss_fn(self, batch, model):
        return compute_reprpo_hra_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.transform
        )


@dataclass(frozen=True)
class ReprPOHRATrainingArguments(TrainingArguments):
    """weights retrain and reroute losses"""
    alpha: int = 0.01

    collection_layers: tuple=(10, 20) 

    """The rank of HRA across different layers. It is best to set 'r' to an even number; otherwise, the default
    initialization method will not work."""
    r: int = 64

    """Whether to apply Gram-Schmidt orthogonalization."""
    apply_GS: bool = False

    _reprpo_class = PL_REPRPO_HRA_MODEL
    _model_keys = ['alpha', 'collection_layers', 'r', 'apply_GS' ]
