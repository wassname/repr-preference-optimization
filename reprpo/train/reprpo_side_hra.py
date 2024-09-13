import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArguments, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict, get_module
from dataclasses import dataclass
import itertools

from .reprpo_hra import HRA, HRA
from .reprpo_side import Sidein, Sideout






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

def mult_with_attention(
    x: Float[Tensor, "b t h"], attn_mask: Float[Tensor, "b t"], dim: int = 1
) -> Float[Tensor, "b t h"]:
    """x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    return (x * layer_attn_mask) / layer_attn_mask.sum(dim, keepdim=True)


# def mean_with_attention(
#     x: Float[Tensor, "b l t h"], attn_mask: Float[Tensor, "b t"], dim: int = 2,
#     eps=1e-12
# ) -> Float[Tensor, "b l h"]:
#     """mean of x, weighted by the attention mask, over dim (token or batch)"""
#     layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
#     return ((x * layer_attn_mask).nansum(dim) + eps) / (
#         layer_attn_mask.nansum(dim) + eps
#     )

# def dist_w_attn_mask(chosen_hs, rejected_hs, attn):
#     dist = rejected_hs - chosen_hs
#     dist = mean_with_attention(dist, attn.detach())
#     assert torch.isfinite(dist).all()  # FIXME nans
#     return (dist**2).nanmean()


def reprpo_forward(model, input_ids, attn_mask, layer_paths, collect_input=True):
    # model.enable_input_require_grads()

    # def make_inputs_require_grad(module, input, output):
    #     output.requires_grad_(True)

    # model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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



def norm(a, dim=-1):
    # return torch.abs(a).mean(dim)
    return torch.pow(a, 2).mean(dim)

def _dist_ratio(a, b, attn, a_ref, b_ref, attn_ref, eps=1e-12) -> Float[Tensor, "b l"]:
    dist = mean_with_attention(a-b, attn)  # reduces over tokens
    dist = norm(dist, dim=-1) # over h
    dist = dist + eps

    dist_ref = mean_with_attention(a_ref-b_ref, attn_ref).detach()
    dist_ref = norm(dist_ref, dim=-1)
    dist_ref = dist_ref + eps

    # get the ratio in log space to avoid div by zero
    log_dist_ratio = torch.log(dist) - torch.log(dist_ref).detach()

    alpha = 1
    return log_dist_ratio * alpha


def dist_ratio(
    a: Dict[str, Float[Tensor, "b t h"]],
    b: Dict[str, Float[Tensor, "b t h"]],
    attn: Float[Tensor, "b t"],
    a_ref: Dict[str, Float[Tensor, "b t h"]],
    b_ref: Dict[str, Float[Tensor, "b t h"]],
    attn_ref: Float[Tensor, "b t"],
) -> float:
    dists = [
        _dist_ratio(a[k], b[k], attn, a_ref[k], b_ref[k], attn_ref) for k in a.keys()
    ]
    # stack each layer now that we've removed the differing h
    d = torch.stack(dists, dim=1)
    return d # reduce(d, "b l -> ", torch.nanmean)


def compute_reprpo_side_hra_loss_batch(
    batch, model, layer_paths, alpha, collect_input, transforms
):
    """Compute the DPO loss on an input batch"""

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = reprpo_forward(
                model=model,
                input_ids=batch["chosen"],
                attn_mask=batch["chosen_mask"],
                layer_paths=layer_paths,
                collect_input=collect_input,
            )
            ref_rej = reprpo_forward(
                model=model,
                input_ids=batch["rejected"],
                attn_mask=batch["rejected_mask"],
                layer_paths=layer_paths,
                collect_input=collect_input,
            )

    model.train()
    pi_cho = reprpo_forward(
        model=model,
        input_ids=batch["chosen"],
        attn_mask=batch["chosen_mask"],
        layer_paths=layer_paths,
        collect_input=collect_input,
    )
    pi_rej = reprpo_forward(
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
    loss_retain = dist_ratio(
        detach_hsd(ref_cho.hs),
        pi_cho.hs,
        comb_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        comb_attn_mask,
    )

    def res_det(hs):
        """use learnable transformation to get the residual part of the hs"""
        return {k: transforms[k](v) for k, v in hs.items()}

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    loss_reroute = dist_ratio(
        res_det(detach_hsd(ref_cho.hs)),
        res_det(pi_rej.hs),
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

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"])
    nll_loss_ratio = nll_loss / ref_nll_loss
    
    with torch.no_grad():
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        # Lets monitor the comparitive norms of the decomposed parts
        def norm_hsd(hs):
            return {k: norm(v) for k, v in hs.items()}
        hs = norm_hsd(ref_cho.hs)
        hs_t = norm_hsd(res_det(ref_cho.hs))
        # hs_resid = norm_hsd(ref_cho.hs - res_det(ref_cho.hs))
        info['hs_t/hs'] = torch.concat([hs_t[k]/hs[k] for k in hs.keys()]).mean()
        # info['hs_resid/hs'] = (hs_resid / hs).mean()
        # info['hs_t/hs_resid'] = (hs_t / hs_resid).mean()

        # also the norm of the weights of the transformation
        info['transform_norm'] = torch.concat([norm(p) for p in transforms.parameters()]).mean()

        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean(),
            nll_loss=nll_loss,
            ref_nll_loss=ref_nll_loss,
            nll_loss_ratio=nll_loss_ratio,
            **info,
        )
    loss = (loss_reroute + loss_retain * alpha).nanmean()
    return loss, info



class PL_REPRPO_SIDE_HRA_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_keys, collection_layers, r, apply_GS, collect_input, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        hra_sizes = {p:get_module(self._model, p).weight.shape[collect_input] for p in self.hparams.layer_paths}
        self.transforms = torch.nn.ParameterDict({k: HRA(dim_hs, dim_hs, r=r, apply_GS=apply_GS) for k,dim_hs in hra_sizes.items()})

    def _loss_fn(self, batch, model):
        return compute_reprpo_side_hra_loss_batch(
            batch,
            model,
            self.hparams.layer_paths,
            self.hparams.alpha,
            collect_input=self.hparams.collect_input,
            transforms=self.transforms
        )


@dataclass(frozen=True)
class SideinHRA(Sidein, HRA):
    r: int = 4

    _reprpo_class = PL_REPRPO_SIDE_HRA_MODEL
    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys', 'r', 'apply_GS']


@dataclass(frozen=True)
class SideoutHRA(SideinHRA, Sideout):
    pass
