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
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss, compute_ptheta
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from ..layers.ether import ETHERLinear, ETHERLinearSmall, _ETHERConfig
from ..layers.hra import HRATransform
# from .reprpo_hra import mean_with_attention, reprpo_forward, dist_ratio
from .reprpo_side import Sidein, Sideout, get_layer_paths, validate_layer_paths
from .reprpo_side_hra import compute_reprpo_side_hra_loss_batch
from baukit.nethook import TraceDict, get_module
from torch.linalg import vecdot
from .reprpo_side import Sidein, Sideout, get_layer_paths, validate_layer_paths, detach_hsd, reprpo_forward_baukit, dist_ratio_dict, mean_with_attention_l
# from .reprpo_hs_dist import loss_dist_along_pref_vect


def loss_dist_along_pref_vect(pi_cho, pi_rej, attn_cho, ref_cho, ref_rej, attn_rej, eps=1e-12) -> Float[Tensor, "b"]:
    """Reward for moving hs along preference vector
    """
    
    pi_cho = mean_with_attention_l(pi_cho, attn_cho)
    pi_rej = mean_with_attention_l(pi_rej, attn_rej)
    ref_cho = mean_with_attention_l(ref_cho, attn_cho)#.detach()
    ref_rej = mean_with_attention_l(ref_rej, attn_rej)#.detach()

    pref_dir = (ref_cho - ref_rej) # preference vector
    cho = pi_cho-ref_cho # vector describing movement of chosen hidden state compared to base model
    rej = pi_rej-ref_rej

    ref_dir_norm = torch.sqrt(torch.linalg.vecdot(pref_dir, pref_dir)).clamp(eps).detach()
    def signed_proj_magnitude(a, ref_dir):
        # get signed projection of `a` along ref_dir
        # like cosine similairy, but without the |a| in the denominator
        a_proj = torch.linalg.vecdot(a, ref_dir, dim=-1) / ref_dir_norm

        # get unsigned length or remainder using pythagorian theorem (we don't care about magnitude here as we )
        a_orth= torch.sqrt(a.pow(2).sum(-1)-a_proj**2)
        angle = F.cosine_similarity(cho, ref_dir, dim=-1)
        # works, but orth gives a nan
        return a_proj, a_orth, angle
    

    signed_cho_proj_pref, cho_orth_pref, cho_cossim = signed_proj_magnitude(cho, pref_dir)
    signed_rej_proj_pref, ref_orth_pref, rej_cossim = signed_proj_magnitude(rej, pref_dir)

    # cho is preferenced over rej, so we expect the hs of cho to stay further along the preference vector than the hs of rej
    is_cho_ahead_of_rej = (torch.linalg.vecdot(pi_cho-pi_rej, pref_dir, dim=-1) / ref_dir_norm)
    if not is_cho_ahead_of_rej.mean()>0:
        warnings.warn("cho should be further than rej")

    # FIXME: make sure cho is further than rej
    # they all start at zero
    β = .1 # factor to punish orthogonal movement

    # if either of the policies hs move along the preference vector that's good, it reduces the loss
    loss_cho_proj = -signed_cho_proj_pref -signed_rej_proj_pref 
    
    # FIXME do I really need this... if a direction is ok with DPO and NLL loss then it should be ok here? plus nan's
    # if they move orthogonal to the preference vector that's bad, it increases the loss
    loss_cho_orth = cho_orth_pref + ref_orth_pref

    # we could also optimize angle, we want it to be close to 1, so we make it negative
    loss_angle =  2 - cho_cossim - rej_cossim
    loss_reroute = (
        loss_cho_proj
         + β  *loss_cho_orth
        + β * loss_angle
    )

    # TODO find better scaling, it needs to be small compared to nll and dpo losses which can be <0.1
    loss = torch.tanh(loss_reroute/3)/10

    return loss, dict(
        loss_cho_proj=loss_cho_proj.mean().detach(),
        loss_cho_orth=loss_cho_orth.mean().detach(),
        loss_angle=loss_angle.mean().detach(),

    )

def dist_ratio_dist(
    a: Dict[str, Float[Tensor, "b t h"]],
    b: Dict[str, Float[Tensor, "b t h"]],
    attn: Float[Tensor, "b t"],
    a_ref: Dict[str, Float[Tensor, "b t h"]],
    b_ref: Dict[str, Float[Tensor, "b t h"]],
    attn_ref: Float[Tensor, "b t"],
) -> float:
    outs = [
        loss_dist_along_pref_vect(a[k], b[k], attn, a_ref[k], b_ref[k], attn_ref) for k in a.keys()
    ]
    dists = [o[0] for o in outs]
    infos = [o[1] for o in outs]
    info = {k: torch.stack([i[k] for i in infos]).mean() for k in infos[0].keys()}
    # stack each layer now that we've removed the differing h
    d = torch.stack(dists, dim=1)
    return d, info

def compute_reprpo_side_dist_loss_batch(batch, model, layer_paths, alpha, collect_input, transforms):

    
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


    def t(hs):
        """use learnable transformation to get the residual part of the hs"""
        return {k: transforms[k](v) for k, v in hs.items()}

    # # loss_retain: more of chosen, less of rejected, on the plane defined by the learnable orthogonal transformation
    # loss_reroute, info_2 = dist_ratio_dist(
    #     t(pi_cho.hs),
    #     t(pi_rej.hs),
    #     cho_attn_mask,
    #     t(ref_cho.hs),
    #     t(ref_rej.hs),
    #     rej_attn_mask,
    # )

    # # loss_reroute: keep chosen the same
    loss_reroute, info_2 = dist_ratio_dist(
        pi_cho.hs,
        pi_rej.hs,
        cho_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        rej_attn_mask,
    )

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss - ref_nll_loss

    # we want to punish it for coherency loss, which nll is a proxy for
    # but not reward it for coherency increase over the reference model
    # as this is not what we are optimising
    loss_nll_retain = F.relu(nll_loss_ratio) #  - math.log(0.9))
    # β = 2
    # loss_nll_retain = (β*nll_loss_ratio - 1) ** 2
    # loss_nll_retain = F.logsigmoid(nll_loss - ref_nll_loss)

    ptheta = compute_ptheta(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )
    
    
    # this loss says: find a transform, where you can make the good hidden states more common and the bad hidden states less common, while not making the outputs incoherent (nll) or favouring the bad response (dpo)
    # TODO if this works tidy it up, and find a way to naturally balance the losses, for example make all in log domain
    loss_dpo_retain = F.relu(-ptheta)
    # loss_reroute = loss_t_reroute
    # loss_retain = loss_dpo
    # loss_retain = loss_nll_retain.mean(1) + loss_dpo_retain
    loss_retain = loss_dpo_retain
    loss = loss_reroute.mean() + alpha * loss_retain.mean()

    def cosine_on_keys(hs1, hs2):
        return torch.stack(
            [
                F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
                for k in hs1.keys()
            ]
        ).nanmean()
    
    with torch.no_grad():
        # get the dpo metrics for comparison
        _, info = compute_dpo_loss(
            pi_cho.logprobs,
            pi_rej.logprobs,
            ref_cho.logprobs,
            ref_rej.logprobs,
        )
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        # # Lets monitor the comparitive norms of the decomposed parts
        # hs = torch.norm(ref_cho.hs)
        # hs_t = torch.norm(t(ref_cho.hs))
        # hs_resid = torch.norm(ref_cho.hs - t(ref_cho.hs))
        # info['hs_t/hs'] = (hs_t / hs).mean()
        # info['hs_resid/hs'] = (hs_resid / hs).mean()
        # info['hs_t/hs_resid'] = (hs_t / hs_resid).mean()

        # # also the norm of the weights of the transformation
        # info['transform_norm'] = sum([torch.norm(p).mean() for p in transform.parameters()])


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean() * alpha,
            nll_loss=nll_loss.mean(),
            ref_nll_loss=ref_nll_loss.mean(),
            nll_loss_ratio=nll_loss_ratio.mean(),
            loss_nll_retain=loss_nll_retain.mean(),
            loss_dpo_retain=loss_dpo_retain.mean(),
            **info_2,
            **info,
        )
    assert torch.isfinite(loss)
    return loss, info

class PL_REPRPO_SIDE_DIST_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, 
                nb, Htype, ether_dropout, flip_side,
                collect_input, collection_layers_side, collection_keys_in: tuple=None, collection_keys_out: tuple=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers_side)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        if collect_input:
            hra_sizes = {p:get_module(self._model, p).in_features for p in self.hparams.layer_paths}
        else:
            hra_sizes = {p:get_module(self._model, p).out_features for p in self.hparams.layer_paths}
        self.transforms = torch.nn.ParameterDict({k: 
                                                  ETHERLinearSmall(
                                                      dim_hs, dim_hs, 
                                                      nb=nb,
                                                      Htype=Htype,
                                                      ether_dropout=ether_dropout,
                                                      flip_side=flip_side,
                                                      reduction=512, # 16k/
                                                      ) for k,dim_hs in hra_sizes.items()})
        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)

    def _loss_fn(self, batch, model):
        return compute_reprpo_side_dist_loss_batch(
            batch,
            model,
            layer_paths=self.hparams.layer_paths,
            alpha=self.hparams.alpha,
            collect_input=self.hparams.collect_input,
            transforms=self.transforms,
        )


@dataclass
class SideDist(_ETHERConfig, TrainingArgumentswCollection):
    """
    Loss: movement of hs along the hs pref vector.
    
    Transform: ETHER  
     
    """

    alpha: float = 10
    """balancing retrain and reroute losses"""

    lr: float = 2e-5

    Htype: str = 'etherplus'

    nb: int = 32

    rel_loss: bool = True

    _reprpo_class = PL_REPRPO_SIDE_DIST_MODEL
    _model_keys = ['alpha', 'collection_layers_side', 'collect_input', 'collection_keys_in', 'nb', 'Htype', 'ether_dropout', 'flip_side', ]

