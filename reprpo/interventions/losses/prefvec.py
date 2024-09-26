from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from .helpers import cross_entropy_loss, compute_ptheta
from ..types import HS, Mask, ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


def prefec_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transform: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps=1e-12,
    β = 0.1,
    use_orth_loss=True,
    use_angle_loss=True,
    use_dpo_loss=True,
    use_nll_loss=True,
    weight_tokens=False,
):
    """
    movement of hs along the hs pref vector.
    """

    def preproc_hs(o, k):
        hs = o.hs[k]
        if transform is not None:
            hs = transform(hs)
        hs = reduce_tokens_w_attention(hs, o.mask, weight_tokens=weight_tokens)
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

        pref_dir_unit = pref_dir / pref_dir.norm(dim=-1, keepdim=True).clamp(eps)

        def signed_proj_magnitude(a, ref_dir):
            # get projection of `a` along ref_dir
            a_proj = (pref_dir_unit * a).sum(dim=-1, keepdim=True) * pref_dir_unit

            a_orth = a - a_proj
            # if torch.norm(a)>0:
            #     # causes NaN's is used with norm(a)==0
            #     # sqrt(a^2 - (a*ref_dir)^2)
            #     # sqrt(a*(a - ref_dir))
            # else:
            #     a_orth = torch.zeros_like(a_proj)
            cosine_sim = F.cosine_similarity(a, ref_dir, dim=-1)
            return a_proj.mean(dim=-1), a_orth.mean(dim=-1), cosine_sim

        cho_pref, cho_orth, cho_cossim = signed_proj_magnitude(
            cho, pref_dir
        )
        rej_pref, ref_orth, rej_cossim = signed_proj_magnitude(
            rej, pref_dir
        )

        # goes down if the hs moves along the direction of the preference vector
        loss_cho_proj = -cho_pref - rej_pref

        # increases with movement of hs orthogonal to the preference vector
        loss_cho_orth = torch.abs(cho_orth) + torch.abs(ref_orth)

        # we could also optimize angle, we want it to be close to 1, so we make it negative
        #  cosine sim ranges from -1 meaning exactly opposite, to 1 meaning exactly the same, with 0 indicating orthogonality
        # so 1-cosine sim ranges from 0 to 2, with 0 meaning exactly the same, and 2 meaning exactly opposite
        loss_angle = 2 - cho_cossim - rej_cossim

        return dict(
            loss_cho_proj=loss_cho_proj,
            loss_cho_orth=loss_cho_orth,
            loss_angle=loss_angle,

            _cho_orthorgonal2pref=cho_orth,
            _ref_orthorgonal2pref=ref_orth,
            _signed_cho_pref=cho_pref,
            _signed_rej_pref=rej_pref,
            _cho_cosine_similarity=cho_cossim,
            _rej_cosine_similarity=rej_cossim,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    loss_reroute = ll["loss_cho_proj"]
    if use_orth_loss:
        loss_reroute += β * ll["loss_cho_orth"]
    if use_angle_loss:
        loss_reroute += β * ll["loss_angle"]
    # TODO find better scaling, it needs to be small compared to nll and dpo losses which can be <0.1
    loss_reroute = torch.tanh(loss_reroute / 3) / 10

    # nll loss, to ensure it's punished for less coherent outputs
    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch["chosen_mask"])
    ref_nll_loss = cross_entropy_loss(
        ref_cho.logits, batch["chosen"], batch["chosen_mask"]
    )
    nll_loss_ratio = (nll_loss - ref_nll_loss).mean(1)
    loss_nll_retain = F.relu(nll_loss_ratio)
    # FIXME why is loss_nll_retain<>nll_loss_ratio is logs?

    # dpo loss, punished model if rejected completion is more likely than the chosen
    ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    loss_dpo_retain = F.relu(-ptheta)

    loss_retain = torch.zeros_like(loss_reroute)
    if use_dpo_loss:
        loss_retain += loss_dpo_retain
    if use_nll_loss:
        loss_retain += loss_nll_retain

    loss = loss_reroute.mean() + α * loss_retain.mean()

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


@dataclass
class PrefVecLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states along the preference vector, with some constraints:
    - keep text at least as coherent (relu(mode/base), 
    - keep the chosen answer at least prefered (relu(rej-cho)
    - punish movement orthogonal to the preference vector: by distance * β
    - punish movement orthogonal to the preference vector: by angle * β
    """

    # remove this as an option since we always want retain to be large
    # α: float = 1.0
    # """balance between reroute and retain loss."""

    eps: float = 1.e-12

    β: float = 1.e-6
    """factor to punish orthogonal movement"""

    use_orth_loss: bool = True
    """punish movement orthogonal to the preference vector: by distance"""

    use_angle_loss: bool = True
    """punish movement orthogonal to the preference vector: by angle"""

    use_dpo_loss: bool = False
    """punish model if rejected completion is more likely than the chosen"""

    use_nll_loss: bool = False
    """punish model if output is less coherent than reference model"""

    weight_tokens: bool = False
    """exp weight tokens along seq"""

    def c(self, *args, **kwargs):
        return prefec_loss(*args, **kwargs, **asdict(self))
