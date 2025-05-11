from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from .helpers import cross_entropy_loss, compute_ptheta
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


def prefvec_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps=1e-12,
    β=0.1,
    use_orth_loss=True,
    use_angle_loss=True,
    use_dpo_loss=True,
    use_nll_loss=True,
    weight_tokens=False,
    use_proj_rel=False,
    use_pref_ref=True,
):
    """
    movement of hs along the hs pref vector.
    """

    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k: str):
        hs = o.hs[k]#.softmax(-1)
        hs = reduce_tokens_w_attention(hs, o.mask, weight_tokens=weight_tokens)
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)  # .detach()
        hs_ref_rej = preproc_hs(ref_rej, k)  # .detach()

        # we define the reference vector as the direction between the reference chosen and rejected hidden states. It's a high dim vector in the space of hidden states
        pref_dir_ref = hs_ref_cho - hs_ref_rej  # preference vector
        pref_dir_pi = hs_pi_cho - hs_pi_rej  # preference vector
        cho = (
            hs_pi_cho - hs_ref_cho
        )  # vector describing movement of chosen hidden state compared to base model
        rej = hs_pi_rej - hs_ref_rej

        # pref_dir_ref_unit = pref_dir_ref / pref_dir_ref.norm(dim=-1, keepdim=True).clamp(eps)
        # pref_dir_pi_unit = pref_dir_pi / pref_dir_pi.norm(dim=-1, keepdim=True).clamp(eps)

        def signed_proj_magnitude(a, pref_dir):
            pref_dir_unit = pref_dir / pref_dir.norm(dim=-1, keepdim=True).clamp(eps)

            # get projection of `a` along ref_dir
            a_proj = (pref_dir_unit * a).sum(dim=-1, keepdim=True) * pref_dir_unit

            a_orth = a - a_proj
            cosine_sim = F.cosine_similarity(a, pref_dir, dim=-1)
            return a_proj.mean(dim=-1), a_orth.mean(dim=-1), cosine_sim

        if use_pref_ref:
            cho_pref, cho_orth, cho_cossim = signed_proj_magnitude(cho, pref_dir_ref)
            rej_pref, ref_orth, rej_cossim = signed_proj_magnitude(rej, pref_dir_ref)
            rel_pref, rel_orth, rel_cossim = signed_proj_magnitude(hs_pi_cho-hs_pi_rej, pref_dir_ref)
        else:
            cho_pref, cho_orth, cho_cossim = signed_proj_magnitude(cho, pref_dir_pi)
            rej_pref, ref_orth, rej_cossim = signed_proj_magnitude(rej, pref_dir_pi)
            rel_pref, rel_orth, rel_cossim = signed_proj_magnitude(hs_pi_cho-hs_pi_rej, pref_dir_pi)

        # goes down if either hs moves along the direction of the preference vector. Note this does not seperate out the preferences
        # TODO do a whole experiment on this line as it was learning!
        loss_proj =  -(cho_pref - rej_pref)

        # we would also like cho to be preferenced over ref, so make them far away?
        # note is at 0 initially and does not move

        # loss_proj_rel = -(cho_pref - rej_pref)
        # TODO do exp on this line
        loss_proj_rel = -rel_pref

        # increases with movement of hs orthogonal to the preference vector
        loss_orth = torch.abs(cho_orth) + torch.abs(ref_orth)

        loss_rel_orth = torch.abs(rel_orth)

        # we could also optimize angle, we want it to be close to 1, so we make it negative
        #  cosine sim ranges from -1 meaning exactly opposite, to 1 meaning exactly the same, with 0 indicating orthogonality
        # so we shift it to be a positive loss than approaches zero
        loss_angle = 3 - cho_cossim + rej_cossim - rel_cossim

        return dict(
            loss_proj=loss_proj,
            loss_orth=loss_orth,
            loss_rel_orth=loss_rel_orth,
            loss_angle=loss_angle,
            loss_proj_rel=loss_proj_rel,
            _cho_orthorgonal2pref=cho_orth,
            _ref_orthorgonal2pref=ref_orth,
            _signed_cho_pref=cho_pref,
            _signed_rej_pref=rej_pref,
            _cho_cosine_similarity=cho_cossim,
            _rej_cosine_similarity=rej_cossim,
            _rel_cosine_similarity=rel_cossim,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    if use_proj_rel:
        loss_reroute = 1+ll["loss_proj_rel"] * 1e5 # a very small number so we make it bigger
    else:
        loss_reroute = 1+ll["loss_proj"] * 1e5

    # tgis beeds balance
    if use_orth_loss:
        if use_proj_rel:
            loss_reroute += β * ll["loss_rel_orth"] * 1e5
        else:
            loss_reroute += β * ll["loss_orth"] * 1e5
    
    if use_angle_loss:
        loss_reroute += β * ll["loss_angle"]
    
    # TODO find better scaling, it needs to be small compared to nll and dpo losses which can be <0.1
    # if use_dpo_loss or use_nll_loss:
    #     loss_reroute = torch.tanh(loss_reroute / 3) / 10

    # nll loss, to ensure it's punished for less coherent outputs
    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch["chosen_mask"])
    ref_nll_loss = cross_entropy_loss(
        ref_cho.logits, batch["chosen"], batch["chosen_mask"]
    )
    nll_loss_ratio = (nll_loss - ref_nll_loss)
    loss_retain_nll = F.relu(nll_loss_ratio).mean(1)
    # FIXME why is loss_retain_mll<>nll_loss_ratio in logs?

    # dpo loss, punished model if rejected completion is more likely than the chosen
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    # normally dpo has logsigmoid, but we might want to only punish not reward, but using relu # https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L82
    # loss_retain_dpo = F.relu(-dpo_ptheta)
    loss_retain_dpo = F.logsigmoid(-dpo_ptheta)

    loss_retain = torch.zeros_like(loss_reroute)
    if use_dpo_loss:
        loss_retain += loss_retain_dpo
    if use_nll_loss:
        loss_retain += loss_retain_nll * 100 # HACKY balance

    def signed_square_loss(loss):
        return torch.sign(loss) * (loss ** 2)

    # loss = signed_square_loss(loss_reroute).mean() + α * signed_square_loss(loss_retain).mean()
    loss = loss_reroute.mean() + α * loss_retain.mean()

    info = dict(
        loss_reroute=loss_reroute,
        loss_dpo_retain=loss_retain_dpo,
        loss_nll_retain=loss_retain_nll,
        loss_retain=loss_retain,
        nll_loss_ratio=nll_loss_ratio,
        ptheta=dpo_ptheta,
        **ll,
    )
    info = {k: v.mean().detach() for k, v in info.items()}

    return loss, info


@dataclass
class PrefVecLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states along the preference vector, with some constraints:
    - keep text at least as coherent (relu(mode/base), (nll_loss)
    - keep the chosen answer at least prefered (relu(rej-cho) dpo_loss
    - punish movement orthogonal to the preference vector: by distance * β
    - punish movement orthogonal to the preference vector: by angle * β
    """

    # α: float = 1.0
    # """balance between reroute and retain loss."""

    eps: float = 1.0e-12

    β: float = 0.5
    """factor to punish orthogonal movement"""

    use_orth_loss: bool = True
    """punish movement orthogonal to the preference vector: by distance"""

    use_angle_loss: bool = True
    """punish movement orthogonal to the preference vector: by angle"""

    use_dpo_loss: bool = True
    """punish model if rejected completion is more likely than the chosen"""

    use_nll_loss: bool = False
    """punish model if output is less coherent than reference model"""

    weight_tokens: bool = False
    """exp weight tokens along seq"""

    use_proj_rel: bool = True
    """encourage chosen to be more in the pref dir than rejected"""

    use_pref_ref: bool = True
    """use the reference model to get the preference vector. Is false we use the moving policy and it could lead to a better outcome, or instability (TODO investigate)"""

    def c(self, *args, **kwargs):
        return prefvec_loss(*args, **kwargs, **asdict(self))
