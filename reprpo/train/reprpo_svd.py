import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.lightning import PL_MODEL, TrainingArguments, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from reprpo.helpers.svd_decomposer import SoftSVDDecomposer, DualSVDDecomposer, SVDDecomposer



@dataclass
class ReprPOSVDTrainingArguments(TrainingArguments):
    """weights retrain and reroute losses"""
    alpha: int = 0.3

    """we decompose the embedded and de-embedding layers using SVD then remove the top <quantile> of singular values from the hidden states"""
    # note removing the 0.25 top singular values removes 90% of the magnitude from hs leaving a small
    quantile: float=1

    """if true, will use the embedding and lm_head, if false only lm_head"""
    dual_svd: bool = False

    adapter_name: str = "reprpo_svd"

    collection_layers: tuple=(10, 12, 14, 16, 18, 20, 22, 24, 26, 28) 

    lr: float = 1e-4

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


def compute_reprpo_svd_loss_batch(batch, model, alpha, collection_layers, decomposer):

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
        """use SVD to decompose hs into the inputoutput and residual components"""
        hs_io = decomposer(hs)
        return hs - hs_io#.detach() # FIXME, should I not detatch this?

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    # start at 0 and go to -ve inf
    # start at log(1)=0 go to log(0)=-inf
    loss_reroute = dist_ratio(
        res_det(ref_cho.hs).detach(),
        res_det(pi_rej.hs),
        comb_attn_mask,
        res_det(ref_cho.hs),
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
        hs = norm(ref_cho.hs)
        hs_r = norm(res_det(ref_cho.hs))
        hs_io = norm(decomposer(ref_cho.hs))
        info['hs_r/hs'] = (hs_r / hs).mean()
        info['hs_io/hs'] = (hs_io / hs).mean()
        info['hs_r/hs_io'] = (hs_r / hs_io).mean()


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

class PL_REPRPO_SVD_MODEL(PL_MODEL):
    def __init__(self, *args, alpha=1, collection_layers=[10, 20], quantile=0.75, dual_svd=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        # convert
        if dual_svd:
            self.decomposer = DualSVDDecomposer(
                self._model.get_input_embeddings().weight.clone().float(),
                self._model.lm_head.weight.clone(),
                quantile=quantile,
            )
        else:
            if quantile < 1:
                self.decomposer = SoftSVDDecomposer(
                    self._model.lm_head.weight.clone().float(), quantile=quantile
                )
            else:
                self.decomposer = SVDDecomposer(
                    self._model.lm_head.weight.clone().float()
                )

    def _loss_fn(self, batch, model):
        return compute_reprpo_svd_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.decomposer,
        )
