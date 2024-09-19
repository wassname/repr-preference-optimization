import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict, get_module
from dataclasses import dataclass
import itertools
from reprpo.train.reprpo_hra import dist_ratio



def get_layer_paths(collection_keys, collection_layers):
    layer_paths = [
        [p.format(layer=layer) for p in collection_keys] for layer in collection_layers
    ]
    layer_paths = list(itertools.chain(*layer_paths))
    return layer_paths


def validate_layer_paths(model, layer_paths):
    for p in layer_paths:
        get_module(model, p)


def detach_hsd(hs):
    return {k: v.detach() for k, v in hs.items()}


def mean_with_attention(
    x: Float[Tensor, "b t h"], attn_mask: Float[Tensor, "b t"], dim: int = 1
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

    return SimpleNamespace(hs=reprs, logits=outs.logits, logprobs=logprobs)


def dist_ratio(a, b, attn, a_ref, b_ref, attn_ref, eps=1e-12) -> Float[Tensor, "b l"]:

    dist = mean_with_attention(a-b, attn)  # reduces over tokens
    dist = torch.norm(dist, dim=-1) # over h
    dist = dist + eps
    log_dist_ratio = torch.log(dist)

    # if provided with reference points, return the distance as a ratio to the reference distance
    if (a_ref is not None) and (b_ref is not None) and (attn_ref is not None):
        dist_ref = mean_with_attention(a_ref-b_ref, attn_ref).detach()
        dist_ref = torch.norm(dist_ref, dim=-1)
        dist_ref = dist_ref + eps

    # get the ratio in log space to avoid div by zero
    log_dist_ratio = torch.log(dist) - torch.log(dist_ref).detach()

    alpha = 1
    return log_dist_ratio * alpha


def dist_ratio_dict(
    a: Dict[str, Float[Tensor, "b t h"]],
    b: Dict[str, Float[Tensor, "b t h"]],
    attn: Float[Tensor, "b t"],
    a_ref: Dict[str, Float[Tensor, "b t h"]],
    b_ref: Dict[str, Float[Tensor, "b t h"]],
    attn_ref: Float[Tensor, "b t"],
) -> float:
    dists = [
        dist_ratio(a[k], b[k], attn, a_ref[k], b_ref[k], attn_ref) for k in a.keys()
    ]
    # stack each layer now that we've removed the differing h
    d = torch.stack(dists, dim=1)
    return d # reduce(d, "b l -> ", torch.nanmean)


def compute_reprpo_side_loss_batch(
    batch, model, layer_paths, alpha, collect_input=True
):
    """Compute the DPO loss on an input batch"""

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = reprpo_forward_baukit(
                model=model,
                input_ids=batch["chosen"],
                attn_mask=batch["chosen_mask"],
                layer_paths=layer_paths,
                collect_input=collect_input,
            )
            ref_rej = reprpo_forward_baukit(
                model=model,
                input_ids=batch["rejected"],
                attn_mask=batch["rejected_mask"],
                layer_paths=layer_paths,
                collect_input=collect_input,
            )

    model.train()
    pi_cho = reprpo_forward_baukit(
        model=model,
        input_ids=batch["chosen"],
        attn_mask=batch["chosen_mask"],
        layer_paths=layer_paths,
        collect_input=collect_input,
    )
    pi_rej = reprpo_forward_baukit(
        model=model,
        input_ids=batch["rejected"],
        attn_mask=batch["rejected_mask"],
        layer_paths=layer_paths,
        collect_input=collect_input,
    )
    cho_attn_mask = batch["chosen_mask"]
    rej_attn_mask = batch["rejected_mask"]

    comb_attn_mask = cho_attn_mask * rej_attn_mask

    # loss_retain: the representation of policy chosen responses should be closer to the reference chosen responses
    # and again we scale it using the reference model as a stable target
    loss_retain = dist_ratio_dict(
        detach_hsd(ref_cho.hs),
        pi_cho.hs,
        comb_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        comb_attn_mask,
    )

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    loss_reroute = dist_ratio_dict(
        detach_hsd(ref_cho.hs),
        pi_rej.hs,
        comb_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        comb_attn_mask,
    )


    # get the dpo metrics for comparison
    _, info = compute_dpo_loss(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )

    
    def cosine_on_keys(hs1, hs2):
        return torch.stack(
            [
                F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
                for k in hs1.keys()
            ]
        ).nanmean()

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss / ref_nll_loss
    
    with torch.no_grad():
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean(),
            nll_loss=nll_loss.mean(),
            ref_nll_loss=ref_nll_loss.mean(),
            nll_loss_ratio=nll_loss_ratio.mean(),
            **info,
        )
    loss = (loss_reroute + loss_retain * alpha).nanmean()
    return loss, info



class PL_REPRPO_SIDE_MODEL(PL_MODEL):
    def __init__(self, *args, alpha: float, collection_layers: list, collect_input: bool, collection_keys_in: list=None, collection_keys_out: list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

    def _loss_fn(self, batch, model):
        return compute_reprpo_side_loss_batch(
            batch,
            model,
            self.hparams.layer_paths,
            self.hparams.alpha,
            collect_input=self.hparams.collect_input,
        )


@dataclass
class Sidein(TrainingArgumentswCollection):
    """
    Target: `layer.out_proj.in`.

    here we collect the **inputs** from the output modules of the each layer.

    in other words we do not collect the contribution to the hidden states but instead activations internal to the layer
    
    """

    _reprpo_class = PL_REPRPO_SIDE_MODEL
    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_in']



@dataclass
class Sideout(TrainingArgumentswCollection):
    """
    Target: `layer.in_proj.out`.

    in other words we do not collect the contribution to the hidden states but instead activations internal to the layer

    collecting the input.outs is often more complex, but baukit sometimes can handle outs better
    """
    collect_input: bool = False


    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_out']

    _reprpo_class = PL_REPRPO_SIDE_MODEL
