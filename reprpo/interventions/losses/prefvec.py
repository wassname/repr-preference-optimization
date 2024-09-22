from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from .helpers import cross_entropy_loss, compute_ptheta
from ..types import HS, Mask, ReprPOModelOutput
from ..reprpo.helpers import mean_tokens_w_attention


def prefec_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transform: Optional[Callable] = None,
    # custom loss_args
    alpha: float = 1.0,
    eps=1e-12,
    β = 0.1,
):
    """
    movement of hs along the hs pref vector.
    """

    def preproc_hs(o, k):
        hs = o.hs[k]
        if transform is not None:
            hs = transform(hs)
        hs = mean_tokens_w_attention(hs, o.mask)
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, 'b']]:
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)  # .detach()
        hs_ref_rej = preproc_hs(ref_rej, k)  # .detach()

        # we define the reference vector as the direction between the reference chosen and rejected hidden states. It's a high dim vector in the space of hidden states
        pref_dir = hs_ref_cho - hs_ref_rej  # preference vector
        cho = (
            hs_pi_cho - hs_ref_cho
        )  # vector describing movement of chosen hidden state compared to base model
        rej = hs_pi_rej - hs_ref_rej

        ref_dir_norm = (
            torch.sqrt(torch.linalg.vecdot(pref_dir, pref_dir)).clamp(eps).detach()
        )

        def signed_proj_magnitude(a, ref_dir):
            # get signed projection of `a` along ref_dir
            # like cosine similairy, but without the |a| in the denominator
            a_proj = torch.linalg.vecdot(a, ref_dir, dim=-1) / ref_dir_norm

            # get unsigned length or remainder using pythagorian theorem (we don't care about magnitude here as we )
            a_orth = torch.sqrt(a.pow(2).sum(-1) - a_proj**2)
            angle = F.cosine_similarity(cho, ref_dir, dim=-1)
            # angle works, but orth gives a nan
            return a_proj, a_orth, angle

        signed_cho_proj_pref, cho_orth_pref, cho_cossim = signed_proj_magnitude(
            cho, pref_dir
        )
        signed_rej_proj_pref, ref_orth_pref, rej_cossim = signed_proj_magnitude(
            rej, pref_dir
        )

        # goes down if the hs moves along the direction of the preference vector
        loss_cho_proj = -signed_cho_proj_pref - signed_rej_proj_pref

        # increases with movement of hs orthogonal to the preference vector
        loss_cho_orth = cho_orth_pref + ref_orth_pref

        # we could also optimize angle, we want it to be close to 1, so we make it negative
        loss_angle = 2 - cho_cossim - rej_cossim

        return dict(
            loss_cho_proj=loss_cho_proj,
            loss_cho_orth=loss_cho_orth,
            loss_angle=loss_angle,

            _cho_orth_pref=cho_orth_pref,
            _ref_orth_pref=ref_orth_pref,
            _signed_cho_proj_pref=signed_cho_proj_pref,
            _signed_rej_proj_pref=signed_rej_proj_pref,
            _cho_cossim=cho_cossim,
            _rej_cossim=rej_cossim,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    loss_reroute = ll["loss_cho_proj"] + β * ll["loss_cho_orth"] + β * ll["loss_angle"]

    # TODO find better scaling, it needs to be small compared to nll and dpo losses which can be <0.1
    loss_reroute = torch.tanh(loss_reroute / 3) / 10

    # nll loss, to ensure it's punished for less coherent outputs
    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch["chosen_mask"])
    ref_nll_loss = cross_entropy_loss(
        ref_cho.logits, batch["chosen"], batch["chosen_mask"]
    )
    nll_loss_ratio = nll_loss - ref_nll_loss
    loss_nll_retain = F.relu(nll_loss_ratio)

    # dpo loss, punished model if rejected completion is more likely than the chosen
    ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    loss_dpo_retain = F.relu(-ptheta)

    loss_retain = loss_dpo_retain  # + loss_nll_retain.mean(1)
    loss = loss_reroute.mean() + alpha * loss_retain.mean()

    info = dict(
        loss_reroute=loss_reroute,
        loss_dpo_retain=loss_dpo_retain,
        loss_nll_retain=loss_nll_retain,
        loss_retain=loss_retain,
        nll_loss_ratio=nll_loss_ratio,
        ptheta=ptheta,
        **ll,
    )
    info = {k: v.mean().detach() for k, v in info.items()}

    return loss, info


@dataclass(frozen=True)
class PrefVecLossConfig:
    alpha: float = 1.0
    eps: float = 1e-12

    β = 0.1
    """factor to punish orthogonal movement"""

    def c(self, *args, **kwargs):
        return prefec_loss(*args, **kwargs, **asdict(self))
