
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass

from .helpers import cross_entropy_loss, compute_ptheta
from ..types import HS, Mask, ReprPOModelOutput, Config
from ..reprpo.helpers import mean_tokens_w_attention


def prefec_loss(pi_cho: ReprPOModelOutput,
            pi_rej: ReprPOModelOutput, 
            ref_cho: ReprPOModelOutput, 
            ref_rej: ReprPOModelOutput, 
            batch: Dict[str, Any],
            transform: Optional[Callable] = None,
            # custom loss_args
            alpha: Float = 1,
            eps=1e-12,
            ):
    """
    movement of hs along the hs pref vector.
    """
    

    def preproc_hs(o):
        if transform is not None:
            hs = transform(o.hs)
        hs = hs.log_softmax(-1)
        hs = mean_tokens_w_attention(hs, o.mask)
        return hs

    hs_pi_cho = preproc_hs(pi_cho)
    hs_pi_rej = preproc_hs(pi_rej)
    hs_ref_cho = preproc_hs(ref_cho)#.detach()
    hs_ref_rej = preproc_hs(ref_rej)#.detach()

    # we define the reference vector as the direction between the reference chosen and rejected hidden states. It's a high dim vector in the space of hidden states
    pref_dir = (hs_ref_cho - hs_ref_rej) # preference vector
    cho = hs_pi_cho-hs_ref_cho # vector describing movement of chosen hidden state compared to base model
    rej = hs_pi_rej-hs_ref_rej


    ref_dir_norm = torch.sqrt(torch.linalg.vecdot(pref_dir, pref_dir)).clamp(eps).detach()
    def signed_proj_magnitude(a, ref_dir):
        # get signed projection of `a` along ref_dir
        # like cosine similairy, but without the |a| in the denominator
        a_proj = torch.linalg.vecdot(a, ref_dir, dim=-1) / ref_dir_norm

        # get unsigned length or remainder using pythagorian theorem (we don't care about magnitude here as we )
        a_orth= torch.sqrt(a.pow(2).sum(-1)-a_proj**2)
        angle = F.cosine_similarity(cho, ref_dir, dim=-1)
        # angle works, but orth gives a nan
        return a_proj, a_orth, angle
    

    signed_cho_proj_pref, cho_orth_pref, cho_cossim = signed_proj_magnitude(cho, pref_dir)
    signed_rej_proj_pref, ref_orth_pref, rej_cossim = signed_proj_magnitude(rej, pref_dir)

    # goes down if the hs moves along the direction of the preference vector
    loss_cho_proj = -signed_cho_proj_pref -signed_rej_proj_pref 
    
    # increases with movement of hs orthogonal to the preference vector
    loss_cho_orth = cho_orth_pref + ref_orth_pref

    # we could also optimize angle, we want it to be close to 1, so we make it negative
    loss_angle =  2 - cho_cossim - rej_cossim


    β = .1 # factor to punish orthogonal movement
    loss_reroute = (
        loss_cho_proj
         + β  *loss_cho_orth
        + β * loss_angle
    )

    # TODO find better scaling, it needs to be small compared to nll and dpo losses which can be <0.1
    loss_reroute = torch.tanh(loss_reroute/3)/10


    # nll loss, to ensure it's punished for less coherent outputs
    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss - ref_nll_loss
    loss_nll_retain = F.relu(nll_loss_ratio)

    # dpo loss, punished model if rejected completion is more likely than the chosen
    ptheta = compute_ptheta(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )
    loss_dpo_retain = F.relu(-ptheta)

    loss_retain = loss_dpo_retain #+ loss_nll_retain.mean(1)
    loss = loss_reroute.mean() + alpha * loss_retain.mean()

    info = dict(
        loss_reroute=loss_reroute,
        loss_dpo_retain=loss_dpo_retain,
        loss_nll_retain=loss_nll_retain,
        loss_retain=loss_retain,

        loss_cho_proj=loss_cho_proj,
        signed_cho_proj_pref=signed_cho_proj_pref,
        signed_rej_proj_pref=signed_rej_proj_pref,

        loss_cho_orth=loss_cho_orth,
        cho_orth_pref=cho_orth_pref,
        ref_orth_pref=ref_orth_pref,

        loss_angle=loss_angle,
        cho_cossim=cho_cossim,
        rej_cossim=rej_cossim,

        nll_loss_ratio=nll_loss_ratio,
        ptheta=ptheta,
    )
    info = {k: v.mean().detach() for k, v in info.items()}


    return loss, info


@dataclass
class PrefVecLossConfig(Config):
    alpha: Float = 1
    eps: Float = 1e-12

    _cls = prefec_loss
