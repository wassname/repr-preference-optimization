from jaxtyping import Float
from typing import Any, Callable, Dict, Optional, Literal
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from ..dpo_helpers import cross_entropy_loss, compute_ptheta, compute_policy_weights
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


def safe_signed_log(x: Tensor, eps: float = 1e-12):
    # preserve the sign, only clamp the magnitude
    sign = x.sign()
    mag  = x.abs()
    return sign * torch.log1p(mag)

def symlog(x: Float[Tensor, "batch"]):
    """Symmetric log function to handle both positive and negative values."""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def safe_log(x: Float[Tensor, "batch"], eps=1e-12):
    """Safe log function to avoid log(0) issues."""
    # return torch.log(x.clamp(min=eps))
    return torch.log(x+eps)

def innerdpo_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps: float = 1e-9,
    β: float = 1,
    use_policy_weights: bool = False,
    inner_policy_weights: bool = False,
    align_method: str = 'para_signed',
    norm_before_reduce: bool = True,
):
    """
    Compute innerDPO loss with various alignment options.

    Args:
        align_method (AlignMethod): alignment metric to use (see top-level AlignMethod doc).
    """
    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k: str):
        """Preprocess hidden states: normalize then aggregate."""

        hs = o.hs[k]  # [batch, seq_len, hidden_dim], RAW ACTIVATIONS
        # hs = F.log_softmax(hs, dim=-1)  # [batch, seq_len, hidden_dim], LOG PROBABILITIES
        if norm_before_reduce:
            hs = F.normalize(hs, p=2, dim=-1)  # [batch, seq_len, hidden_dim], UNIT VECTORS. If we normalise before transforms, we get NaNs in the gradients
        # Aggregate over sequence using attention masks
        hs = reduce_tokens_w_attention(hs, o.mask)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k):
        # Get raw hidden states
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)
        hs_ref_rej = preproc_hs(ref_rej, k)

        pref_dir_ref = hs_ref_cho - hs_ref_rej
        pref_dir_pi = hs_pi_cho - hs_pi_rej

        # Decompose pi into parallel and orthogonal components
        pref_dir_ref_unit = F.normalize(pref_dir_ref, p=2, dim=-1)
        # para_sign_magn = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1, keepdim=True)

        para_vec = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1, keepdim=True) * pref_dir_ref_unit
        ort_vec = pref_dir_pi - para_vec

        par_vec_ref = torch.sum(pref_dir_ref * pref_dir_ref_unit, dim=-1, keepdim=True) * pref_dir_ref_unit
        orth_vec_ref = pref_dir_ref - par_vec_ref

        # Magnitudes
        par_mag = torch.norm(para_vec, p=1, dim=-1)
        ort_mag = torch.norm(ort_vec, p=1, dim=-1)
        log_par = safe_signed_log(par_mag, eps=eps)
        log_ort = safe_signed_log(ort_mag, eps=eps)
        logodds_pi = log_par - log_ort

        par_ref = torch.norm(par_vec_ref, p=1, dim=-1)
        ort_ref = torch.norm(orth_vec_ref, p=1, dim=-1)
        log_par_ref = safe_signed_log(par_ref, eps=eps)
        log_ort_ref = safe_signed_log(ort_ref, eps=eps)
        logodds_ref = log_par_ref - log_ort_ref

        # Make weights similar to # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
        scores = torch.stack([log_par, log_ort], dim=-1)    # shape [B,2]
        log_hidden_probs = F.log_softmax(scores, dim=-1)   # [B,2]
        w_adj = torch.logsumexp(2 * log_hidden_probs, dim=-1)  # [B]
        pseudo_lp = log_hidden_probs[...,0] - w_adj     # “chosen” class
        hidden_weight = torch.tensor(1.0, device=pref_dir_pi.device)
        _hidden_weight = torch.clamp(torch.exp(pseudo_lp), max=1.0).detach()  # [B]

        match align_method:
            case 'para_signed':
                """            
                Direct signed projection ⟨policy_diff, ref_unit⟩ 
                Geometric: "How far does policy move along the reference preference direction?"
                Intuition: Raw alignment signal. +ve = aligned, -ve = anti-aligned. Unbounded.
                """
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                hidden_ptheta =  signed_proj
            case 'para_signed_log':
                """                
                sign(projection) * log(|projection|)
                Geometric: Compresses projection magnitude while preserving direction
                Intuition: Stabilizes wild swings, dampens near-zero gradients
                """
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                hidden_ptheta =  safe_signed_log(signed_proj, eps=eps)
            case 'para_orth_signed':
                """
                signed_projection - orthogonal_magnitude
                Hybrid scale: raw signed parallel distance vs raw orthogonal drift
                Geometric: "Raw signed parallel strength vs orthogonal drift"
                Intuition: Rewards ref-direction movement, penalizes off-axis drift (no logs)
                """
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=1, dim=-1)
                hidden_ptheta =  (signed_proj - orthogonal_mag)
            case 'para_orth_signed_log':
                """
                signed_log(parallel) - log(orthogonal_magnitude)
                Full log-space: signed log of parallel vs log of orthogonal drift
                Geometric: "Log-parallel strength vs log-orthogonal drift"
                Intuition: Stabilizes both signals in log space
                """
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=1, dim=-1)
                hidden_ptheta =  (safe_signed_log(signed_proj, eps=eps) - safe_log(orthogonal_mag, eps))
            case 'logodds':
                """ 
                signed_log(parallel) - log(orthogonal)
                Geometric: Same as #3 but with log-stabilized parallel term
                Intuition: Double stabilization for both parallel and orthogonal terms
                """
                hidden_ptheta =  (logodds_pi - logodds_ref)
            case 'logodds_noref':
                hidden_ptheta = logodds_pi
            case 'odds_noref':
                hidden_ptheta= par_mag - ort_mag
            case 'stabilized_ratio':
                """
                Ratio with stabilized denominator to prevent exploitation
                """
                # Use batch statistics for threshold
                ort_floor = torch.max(ort_mag.mean() * 0.1, torch.tensor(eps, device=ort_mag.device))
                stabilized_ort = torch.clamp(ort_mag, min=ort_floor)
                hidden_ptheta = par_mag / stabilized_ort
            case 'cosine_policy_margin':
                """
                cos(pi_cho, pi_rej) - cos(ref_cho, ref_rej)
                Geometric: "Policy's internal separability vs reference's separability"  
                Intuition: Maximize policy's own chosen/rejected margin. Bounded [-2,2]
                """
                hidden_ptheta =  (
                    F.cosine_similarity(hs_pi_cho, hs_pi_rej, dim=-1)
                    - F.cosine_similarity(hs_ref_cho, hs_ref_rej, dim=-1)
                )
            case 'cosine_cross_model':
                """
                cos(pi_cho, ref_cho) - cos(pi_rej, ref_rej)
                Geometric: "Pull policy states toward corresponding reference states"
                Intuition: Explicit state-to-state mimicry, not just directional alignment
                (probobly wont work as we want to go beyond internal representation alignment)
                """
                hidden_ptheta =  (
                    F.cosine_similarity(hs_pi_cho, hs_ref_cho, dim=-1)
                    - F.cosine_similarity(hs_pi_rej, hs_ref_rej, dim=-1)
                )
            case _:
                raise ValueError(f"Unsupported align_method: {align_method}")
        
        # Apply DPO-style loss
        loss_hidden_dpo = -F.logsigmoid(β * hidden_ptheta)
        if inner_policy_weights: # I'm not sure if this is helping
            loss_hidden_dpo = loss_hidden_dpo * hidden_weight
        
        return dict(loss_hidden_dpo=loss_hidden_dpo, hidden_weight=_hidden_weight, hidden_ptheta=hidden_ptheta)


    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    loss_dpo = -F.logsigmoid(β * dpo_ptheta)

    loss = loss_dpo + α * ll['loss_hidden_dpo']
    
    # Apply policy weights if requested
    policy_weights = compute_policy_weights(pi_cho, pi_rej)
    ll['policy_weights'] = policy_weights.mean()
    ll['cho_log_policy_weights'] = torch.exp(pi_cho.log_policy_weights).mean()
    ll['rej_log_policy_weights'] = torch.exp(pi_rej.log_policy_weights).mean()   
    if use_policy_weights:
        loss = loss * policy_weights.detach()

    ll = {k:v.mean() for k, v in ll.items()}
    info = dict(
        loss_dpo=loss_dpo.mean(),
        dpo_ptheta=dpo_ptheta.mean(),
        **ll,
    )

    return loss.mean(), info


@dataclass
class InnerDPOLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states apart along the preference vector, with some constraints, while also doing DPO on outpts
    """

    α: float = 0.1
    """balance between reroute and retain loss."""

    eps: float = 1.0e-9

    β: float = 1.
    """factor to punish orthogonal movement"""

    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""

    inner_policy_weights: bool = False
    """Whether to compute policy weights for the inner DPO loss."""


    align_method: str = 'para_signed'
    """Method to compute alignment between chosen and rejected hidden states."""

    norm_before_reduce: bool = True
    """Whether to normalize hidden states before reducing them to a single vector."""
   

    def c(self, *args, **kwargs):
        return innerdpo_loss(*args, **kwargs, **asdict(self))
