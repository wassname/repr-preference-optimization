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

from ..layers.hra import HRATransform
from .reprpo_hra import HRA, norm
from .reprpo_side import Sidein, Sideout, get_layer_paths, validate_layer_paths, detach_hsd, reprpo_forward, dist_ratio



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
    def __init__(self, *args, alpha, collection_layers, r, apply_GS, collect_input, collection_keys_in: list=None, collection_keys_out: list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        # we do one learnable householder roation per layer
        if collect_input:
            hra_sizes = {p:get_module(self._model, p).in_features for p in self.hparams.layer_paths}
        else:
            hra_sizes = {p:get_module(self._model, p).out_features for p in self.hparams.layer_paths}
        self.transforms = torch.nn.ParameterDict({k: HRATransform(dim_hs, dim_hs, r=r, apply_GS=apply_GS) for k,dim_hs in hra_sizes.items()})
        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)
        # TODO check dtype etc

    def _loss_fn(self, batch, model):
        return compute_reprpo_side_hra_loss_batch(
            batch,
            model,
            self.hparams.layer_paths,
            self.hparams.alpha,
            collect_input=self.hparams.collect_input,
            transforms=self.transforms
        )


@dataclass
class SideinHRA(Sidein, HRA):
    """Transform: HRA. Target: activations from layer.out.input
    """

    r: int = 8
    """rank"""

    alpha: float = 0.001
    """weights retrain and reroute losses"""

    _reprpo_class = PL_REPRPO_SIDE_HRA_MODEL
    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_in', 'r', 'apply_GS']


@dataclass
class SideoutHRA(Sideout, HRA):
    """Transform: HRA. Target: activations from layer.in.output."""

    alpha: float = 0.001

    r: int = 8
    """rank"""

    _reprpo_class = PL_REPRPO_SIDE_HRA_MODEL
    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_out', 'r', 'apply_GS']
