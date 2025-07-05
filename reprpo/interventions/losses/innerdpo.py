from jaxtyping import Float
from typing import Any, Callable, Dict, Optional, Literal
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from ..dpo_helpers import cross_entropy_loss, compute_ptheta, compute_policy_weights, compute_mallows_weights
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


def safe_norm(x: Float[Tensor, "batch"], p: int = 2, dim: int = -1, eps: float = 1e-9):
    """
    Safe norm function to avoid division by zero.
    Returns a tensor with the same shape as x, where norms are clamped to eps.
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)  # Avoid division by zero

def innerdpo_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps: float = 1e-6,
    β: float = 1,
    use_wpo: bool = False,
    use_inner_weights: bool = False,
    align_method: str = 'para_signed',
    norm_before_reduce: bool = True,
    filter_sinks: bool = False,
    trust_region: float = 2.0,
    dpo_loss: str = "ipo",
    p=2,
    label_smoothing=0,
    clamp_bottom: bool = False,
    detach_ref: bool = True,
    use_token_constraint: bool = False,
    use_mallows: bool = False,
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
        # # hs = F.log_softmax(hs, dim=-1)  # [batch, seq_len, hidden_dim], LOG PROBABILITIES
        # if norm_before_reduce:
        #     # why would we do this? to prevent attention sinks. consider clamping 3*std mags
        #     hs = safe_norm(hs, p=p, dim=-1, eps=eps)  # [batch, seq_len, hidden_dim], UNIT VECTORS. If we normalise before transforms, we get NaNs in the gradients

        return hs 
    

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k):
        # Get raw hidden states
        hs_pi_cho_t = preproc_hs(pi_cho, k)
        hs_pi_rej_t = preproc_hs(pi_rej, k)
        if detach_ref:
            hs_pi_rej_t = hs_pi_rej_t.detach() # We want to change cho not rej
        hs_ref_cho_t = preproc_hs(ref_cho, k).detach()  # Don't let reference change of course
        hs_ref_rej_t = preproc_hs(ref_rej, k).detach()  # Don't let reference change of course

        if use_mallows:
            hs_pi_cho_t = compute_mallows_weights(hs_pi_cho_t, pi_cho.mallows_weights)
            hs_pi_rej_t = compute_mallows_weights(hs_pi_rej_t, pi_rej.mallows_weights)
            hs_ref_cho_t = compute_mallows_weights(hs_ref_cho_t, ref_cho.mallows_weights, ref_cho.mask)
            hs_ref_rej_t = compute_mallows_weights(hs_ref_rej_t, ref_rej.mallows_weights, ref_rej.mask)
            # cho_token_deviations = cho_token_deviations[:, :-1]
            # rej_token_deviations = rej_token_deviations[:, :-1]

        # Per-token deviations from reference (L2 distance)
        cho_token_deviations = torch.norm(hs_pi_cho_t - hs_ref_cho_t, p=p, dim=-1)  # [batch, seq_len]
        rej_token_deviations = torch.norm(hs_pi_rej_t - hs_ref_rej_t, p=p, dim=-1)  # [batch, seq_len]


        if use_wpo:
            # can be wpo, or mallows
            policy_weights = compute_policy_weights(pi_cho, pi_rej)
            hs_pi_cho_t = hs_pi_cho_t[:, :-1, :] * policy_weights.unsqueeze(2).detach()
            hs_pi_rej_t = hs_pi_rej_t[:, :-1, :] * policy_weights.unsqueeze(2).detach()
            pi_cho.mask = pi_cho.mask[:, :-1]
            pi_rej.mask = pi_rej.mask[:, :-1]
            cho_token_deviations = cho_token_deviations[:, :-1]
            rej_token_deviations = rej_token_deviations[:, :-1]


        # Aggregate per response (like TDPO aggregates KL)
        cho_total_deviation = reduce_tokens_w_attention(cho_token_deviations.unsqueeze(-1), pi_cho.mask).squeeze(-1)  # [batch]
        rej_total_deviation = reduce_tokens_w_attention(rej_token_deviations.unsqueeze(-1), pi_rej.mask).squeeze(-1)  # [batch]

        hs_pi_cho = reduce_tokens_w_attention(hs_pi_cho_t, pi_cho.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        hs_pi_rej = reduce_tokens_w_attention(hs_pi_rej_t, pi_rej.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        hs_ref_cho = reduce_tokens_w_attention(hs_ref_cho_t, ref_cho.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        hs_ref_rej = reduce_tokens_w_attention(hs_ref_rej_t, ref_rej.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS

        pref_dir_ref = hs_ref_cho - hs_ref_rej
        pref_mag_ref = torch.norm(pref_dir_ref, p=p, dim=-1)
        pref_dir_pi = hs_pi_cho - hs_pi_rej

        # Decompose pi into parallel and orthogonal components
        pref_dir_ref_unit = safe_norm(pref_dir_ref, p=p, dim=-1, eps=eps).detach()

        signed_proj = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)
        para_vec = signed_proj.unsqueeze(1) * pref_dir_ref_unit
        ort_vec = pref_dir_pi - para_vec

        signed_proj_ref = torch.sum(pref_dir_ref * pref_dir_ref_unit, dim=-1).detach()
        par_vec_ref = signed_proj_ref.unsqueeze(1) * pref_dir_ref_unit
        # orth_vec_ref = pref_dir_ref - par_vec_ref

        # Magnitudes
        par_mag = torch.norm(para_vec, p=p, dim=-1)
        ort_mag = torch.norm(ort_vec, p=p, dim=-1)
        log_par = safe_log(par_mag, eps=eps)
        log_ort = safe_log(ort_mag, eps=eps)
        logodds_pi = log_par - log_ort

        par_mag_ref = torch.norm(par_vec_ref, p=p, dim=-1)
        # ort_mag_ref = torch.norm(orth_vec_ref, p=p, dim=-1) # this is often near zero, so we can't use
        log_par_ref = safe_log(par_mag_ref, eps=eps)
        # log_ort_ref = safe_log(ort_mag_ref, eps=eps) # this is often near zero, so we can't use
        # logodds_ref = log_par_ref - log_ort_ref

        # Make weights similar to # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
        scores = torch.stack([log_par, log_ort], dim=-1)    # shape [B,2]
        log_hidden_probs = F.log_softmax(scores, dim=-1)   # [B,2]
        w_adj = torch.logsumexp(2 * log_hidden_probs, dim=-1)  # [B]
        pseudo_lp = log_hidden_probs[...,0] - w_adj     # “chosen” class
        hidden_weight = torch.clamp(torch.exp(pseudo_lp), max=1.0).detach()  # [B]

        match align_method:
            case 'pars_rat':
                hidden_ptheta =  signed_proj / (pref_mag_ref + eps) - 1
            case 'pars_rat_log':
                hidden_ptheta =  10 * (safe_log(signed_proj) - safe_log(pref_mag_ref))
            case 'para_signed':
                """            
                Direct signed projection ⟨policy_diff, ref_unit⟩ 
                Geometric: "How far does policy move along the reference preference direction?"
                Intuition: Raw alignment signal. +ve = aligned, -ve = anti-aligned. Unbounded.
                """
                hidden_ptheta =  signed_proj
            # case 'para_signed_log':
            #     """                
            #     sign(projection) * log(|projection|)
            #     Geometric: Compresses projection magnitude while preserving direction
            #     Intuition: Stabilizes wild swings, dampens near-zero gradients
            #     """
            #     hidden_ptheta =  safe_signed_log(signed_proj, eps=eps)
            # case 'para_orth_signed':
            #     """
            #     signed_projection - orthogonal_magnitude
            #     Hybrid scale: raw signed parallel distance vs raw orthogonal drift
            #     Geometric: "Raw signed parallel strength vs orthogonal drift"
            #     Intuition: Rewards ref-direction movement, penalizes off-axis drift (no logs)
            #     """
            #     orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=p, dim=-1)
            #     hidden_ptheta =  (signed_proj - orthogonal_mag)
            # case 'para_orth_signed_log':
            #     """
            #     signed_log(parallel) - log(orthogonal_magnitude)
            #     Full log-space: signed log of parallel vs log of orthogonal drift
            #     Geometric: "Log-parallel strength vs log-orthogonal drift"
            #     Intuition: Stabilizes both signals in log space
            #     """
            #     orthogonal_mag = torch.norm(pref_dir_pi - signed_proj.unsqueeze(-1) * pref_dir_ref_unit, p=p, dim=-1)
            #     hidden_ptheta =  (safe_signed_log(signed_proj, eps=eps) - safe_log(orthogonal_mag, eps))
            # case 'logodds_noref':
            #     hidden_ptheta = logodds_pi * 10
            # case 'odds_noref':
            #     hidden_ptheta= par_mag - ort_mag
            case _:
                raise ValueError(f"Unsupported align_method: {align_method}")
        
        # Apply DPO-style loss
        if use_inner_weights: # I'm not sure if this is helping
            hidden_ptheta = hidden_ptheta * hidden_weight


        assert hidden_ptheta.shape[0] == pi_cho.hs[k].shape[0], \
            f"hidden_ptheta shape {hidden_ptheta.shape} does not match batch size {pi_cho.hs[k].shape[0]}"
        assert len(hidden_ptheta.shape) == 1

        
        return dict(hidden_weight=hidden_weight, hidden_ptheta=hidden_ptheta,
                    signed_proj=signed_proj, #hptheta_bottom=pref_mag_ref,
                            cho_position_deviation=cho_total_deviation,
                    rej_position_deviation=rej_total_deviation)


    # compute losses per layer
    layer_vals = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(layer_vals.values())).keys()
    layer_vals = {k: torch.stack([v[k] for v in layer_vals.values()], -1) for k in ll_keys}
    
    # Combine layer losses
    # if normalize_layers:
    #     # Simple LayerNorm approach - normalizes across layers for each batch element
    #     for k in ll_keys:
    #         layer_values = llr[k]  # [batch, num_layers]
    #         # Apply layer normalization across the layer dimension
    #         normalized = F.layer_norm(layer_values, (layer_values.size(-1),), eps=eps)
    #         llr[k] = normalized
    
    vals = {k: v.mean(-1) for k, v in layer_vals.items()}  # average over layers

    hidden_ptheta = layer_vals['hidden_ptheta']
    if trust_region > 0:
        # Moving too far in the hidden weight space leads to incoherent, so we will bound the loss to only reward within a trusted region.
        # This is similar to the trust region in PPO.
        # It's defined in the ptheta space, so it depends on the loss, but for `pars_rat` it means it can only get reward for seperating the chosen and rejected hidden states by `trust_region` times the distance in the reference model
        if clamp_bottom:
            hidden_ptheta = torch.clamp(hidden_ptheta, trust_region, None)
        else:
            hidden_ptheta = torch.clamp(hidden_ptheta, None, trust_region)
    
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )

    token_constraint = torch.clamp(layer_vals['rej_position_deviation'] - layer_vals['cho_position_deviation'], min=0)
    if use_token_constraint:
        # This is like in https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/blob/master/trainers.py#L151
        # Intuition: Both responses should deviate similarly from their references. If one is changing much more than the other, adjust our confidence accordingly.
        # token_constraint = layer_vals['rej_position_deviation'] - layer_vals['cho_position_deviation']
        # OR Only penalize when rejected deviates more (prevent sneaky gaming)
        alpha_token = 0.5
        hidden_ptheta = hidden_ptheta - alpha_token * token_constraint



    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    if dpo_loss == "ipo":
        loss_dpo = (dpo_ptheta - 1/(2 * β)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        loss_hidden_dpo = (hidden_ptheta.mean(1) - 1/(2 * β)) ** 2
    elif dpo_loss == "SimPER":
        # https://github.com/tengxiao1/SimPER/blob/main/scripts/simper_trainer.py#L588
        loss_dpo = pi_rej.label_logprobs.exp()-pi_cho.label_logprobs.exp()
        # loss_hidden_dpo = layer_vals['signed_proj'].mean(1) # hmm maybe this should be symlog?
        loss_hidden_dpo = symlog(layer_vals['signed_proj']).mean(1) # hmm maybe this should be symlog?
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        loss_dpo = -F.logsigmoid(β * dpo_ptheta) * (1 - label_smoothing) - F.logsigmoid(-β * dpo_ptheta) * label_smoothing 
        loss_hidden_dpo = -F.logsigmoid(β * hidden_ptheta.mean(1)) - F.logsigmoid(-β * hidden_ptheta.mean(1)) * label_smoothing

    loss = loss_dpo + α * loss_hidden_dpo



    vals = {k:v.mean() for k, v in vals.items()}
    info = dict(
        loss_dpo=loss_dpo.mean(),
        dpo_ptheta=dpo_ptheta.mean(),
        loss_hidden_dpo=loss_hidden_dpo.mean(),
        token_constraint=token_constraint.mean(),
        **vals,
    )

    return loss.mean(), info


@dataclass
class InnerDPOLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states apart along the preference vector, with some constraints, while also doing DPO on outpts
    """

    α: float = 0.1
    """balance between reroute and retain loss."""

    filter_sinks: bool = False
    """Whether to filter attention sinks in the hidden states."""

    eps: float = 1.0e-4

    p: int = 2
    """norm to use for the hidden states, 2 is euclidean, 1 is manhattan"""

    use_wpo: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""

    use_mallows: bool = False
    """Whether to use Mallows weights. If True, the reference model must have `--calc-mallows` flag set."""

    use_inner_weights: bool = False
    """Whether to compute policy weights for the inner DPO loss."""

    use_mallows: bool = False
    """Whether to use Mallows weights"""

    dpo_loss: Literal["dpo", "ipo", "SimPER"] = "SimPER"   

    # The below ones are used by IPO by not SimPER
    trust_region: float = 0
    """prevent over optimising inner states. not used in SimPER"""

    β: float = 0.4
    """Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in the [paper](https://huggingface.co/papers/2310.12036).

    Note 0.1 is good for DPO, 0.4 for IPO see https://huggingface.co/blog/pref-tuning not used in SimPER
    """

    align_method: str = 'pars_rat'
    """Method to compute alignment between chosen and rejected hidden states."""

    clamp_bottom: bool = False

    use_token_constraint: bool = False
    """Whether to use the token constraint to adjust the hidden ptheta. This is like in TDPO https://arxiv.org/abs/2404.11999"""
    # FIXME also add to outer DPO

    detach_ref: bool = False
    """Whether to detach the reference hidden states from the computation graph. This is useful to prevent the reference model from changing during training, which is less desired as it doesn't actually come up during generation"""

    # use_dpo_loss: bool = True

    def c(self, *args, **kwargs):
        return innerdpo_loss(*args, **kwargs, **asdict(self))
