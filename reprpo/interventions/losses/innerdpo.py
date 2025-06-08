from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
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
    mag  = x.abs().clamp(min=eps)
    return sign * torch.log(mag)

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
    eps=1e-6,
    β=1,
    use_policy_weights: bool = False,
    inner_policy_weights: bool = False,
    align_method: str = 'direct_projection',
    norm_before_reduce: bool = True,
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

        # Compute similarity scores (like logits in DPO)
        cho_score = F.cosine_similarity(hs_pi_cho, hs_ref_cho, dim=-1).abs()  # How similar chosen is to ref_chosen
        rej_score = F.cosine_similarity(hs_pi_rej, hs_ref_rej, dim=-1).abs()  # How similar rejected is to ref_rejected

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
        par_pi = torch.norm(para_vec, p=1, dim=-1)
        ort_mag = torch.norm(ort_vec, p=1, dim=-1)
        log_par = safe_signed_log(par_pi, eps=eps)
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
                # Direct signed projection: raw directional alignment signal
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                hidden_ptheta = β * signed_proj
            case 'para_signed_log':
                """
                sign(projection) * log(|projection|)
                Geometric: Compresses projection magnitude while preserving direction
                Intuition: Stabilizes wild swings, dampens near-zero gradients
                """
                # Log-stabilized signed projection
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                hidden_ptheta = β * safe_signed_log(signed_proj, eps=eps)
            case 'para_orth_signed':
                # Log-odds: signed parallel vs orthogonal drift
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=1, dim=-1)
                hidden_ptheta = β * (signed_proj - torch.log(orthogonal_mag + eps))
            case 'para_orth_signed_log':
                """
                signed_projection - log(orthogonal_magnitude)  
                Geometric: "Parallel strength vs orthogonal drift" in log-odds space
                Intuition: Rewards ref-direction movement, penalizes off-axis drift
                """
                # Double-stabilized log-odds  
                signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
                orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=1, dim=-1)
                hidden_ptheta = β * (safe_signed_log(signed_proj, eps=eps) - torch.log(orthogonal_mag + eps))
            case 'logodds':
                """ 
                signed_log(parallel) - log(orthogonal)
                Geometric: Same as #3 but with log-stabilized parallel term
                Intuition: Double stabilization for both parallel and orthogonal terms
                """
                # Reference-normalized log-odds comparison
                hidden_ptheta = β * (logodds_pi - logodds_ref)
            case 'cosine_policy_margin':
                """
                cos(pi_cho, pi_rej) - cos(ref_cho, ref_rej)
                Geometric: "Policy's internal separability vs reference's separability"  
                Intuition: Maximize policy's own chosen/rejected margin. Bounded [-2,2]
                """
                # Policy's internal margin vs reference's margin
                hidden_ptheta = β * (
                    F.cosine_similarity(hs_pi_cho, hs_pi_rej, dim=-1)
                    - F.cosine_similarity(hs_ref_cho, hs_ref_rej, dim=-1)
                )
            case 'cosine_cross_model':
                """
                cos(pi_cho, ref_cho) - cos(pi_rej, ref_rej)
                Geometric: "Pull policy states toward corresponding reference states"
                Intuition: Explicit state-to-state mimicry, not just directional alignment
                """
                # Cross-model state alignment
                hidden_ptheta = β * (
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

    α: float = 0.25
    """balance between reroute and retain loss."""

    eps: float = 1.0e-5

    β: float = 1.
    """factor to punish orthogonal movement"""

    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""

    inner_policy_weights: bool = False
    """Whether to compute policy weights for the inner DPO loss."""


    align_method: str = 'para'
    """Method to compute alignment between chosen and rejected hidden states."""

    norm_before_reduce: bool = True
    """Whether to normalize hidden states before reducing them to a single vector."""
   

    def c(self, *args, **kwargs):
        return innerdpo_loss(*args, **kwargs, **asdict(self))
