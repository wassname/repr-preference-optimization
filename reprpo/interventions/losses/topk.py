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


def safe_norm(x: Float[Tensor, "batch"], p: int = 2, dim: int = -1, eps: float = 1e-9):
    """
    Safe norm function to avoid division by zero.
    Returns a tensor with the same shape as x, where norms are clamped to eps.
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)  # Avoid division by zero

def topk_loss(
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
    use_policy_weights: bool = False,
    inner_policy_weights: bool = False,
    align_method: str = 'para_signed',
    norm_before_reduce: bool = True,
    filter_sinks: bool = True,
    trust_region: float = 2.0,
    dpo_loss: str = "ipo",
    p=2,
    label_smoothing=0,
    clamp_bottom: bool = False,
    detach_ref: bool = True,
    use_token_constraint: bool = True,
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

        hs = o.hs[k]
        return hs 
    

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k):
        # Get raw hidden states
        hs_pi_cho_t = preproc_hs(pi_cho, k)
        hs_pi_rej_t = preproc_hs(pi_rej, k)
        if detach_ref:
            hs_pi_rej_t = hs_pi_rej_t.detach() # We want to change cho not rej
        hs_ref_cho_t = preproc_hs(ref_cho, k).detach()  # Don't let reference change of course
        hs_ref_rej_t = preproc_hs(ref_rej, k).detach()  # Don't let reference change of course

        # Per-token deviations from reference (L2 distance)
        cho_token_deviations = torch.norm(hs_pi_cho_t - hs_ref_cho_t, p=p, dim=-1)  # [batch, seq_len]
        rej_token_deviations = torch.norm(hs_pi_rej_t - hs_ref_rej_t, p=p, dim=-1)  # [batch, seq_len]

        # Aggregate per response (like TDPO aggregates KL)
        cho_total_deviation = reduce_tokens_w_attention(cho_token_deviations.unsqueeze(-1), pi_cho.mask).squeeze(-1)  # [batch]
        rej_total_deviation = reduce_tokens_w_attention(rej_token_deviations.unsqueeze(-1), pi_rej.mask).squeeze(-1)  # [batch]

        hs_pi_cho = reduce_tokens_w_attention(hs_pi_cho_t, pi_cho.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        hs_pi_rej = reduce_tokens_w_attention(hs_pi_rej_t, pi_rej.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        # hs_ref_cho = reduce_tokens_w_attention(hs_ref_cho_t, ref_cho.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        # hs_ref_rej = reduce_tokens_w_attention(hs_ref_rej_t, ref_rej.mask, filter_sinks=filter_sinks)  # [batch, hidden_dim], AVERAGED UNIT VECTORS

        diff = hs_pi_cho - hs_pi_rej

        # Find top-k dimensions by absolute difference
        # TODO k to param
        topk_values, topk_indices = torch.topk(diff.abs(), k=600, dim=-1)
        
        # Create sparse version
        sparse_diff = torch.zeros_like(diff)
        sparse_diff.scatter_(-1, topk_indices, topk_values * diff.sign().gather(-1, topk_indices))
        
        # Loss encourages large separation in these k dimensions
        separation_loss = -sparse_diff.abs()#.mean()
        
        # Regularize to prevent too extreme values
        # TODO this const to param
        extreme_penalty = F.relu(sparse_diff.abs() - 2.0)#.mean()
        
        inner_loss = separation_loss + 0.1 * extreme_penalty
        return dict(inner_loss=inner_loss,
                    
                    sparse_diff_abs=sparse_diff.abs().mean(),
                    separation_loss=separation_loss.mean(),
                    extreme_penalty=extreme_penalty.mean(),


                            cho_position_deviation=cho_total_deviation,
                    rej_position_deviation=rej_total_deviation)


    # compute losses per layer
    layer_vals = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(layer_vals.values())).keys()
    layer_vals = {k: torch.stack([v[k] for v in layer_vals.values()], -1) for k in ll_keys}

    vals = {k: v.mean(-1) for k, v in layer_vals.items()}  # average over layers

    loss = layer_vals['inner_loss']

    token_constraint = torch.clamp(layer_vals['rej_position_deviation'] - layer_vals['cho_position_deviation'], min=0)
    if use_token_constraint:
        # This is like in https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/blob/master/trainers.py#L151
        # Intuition: Both responses should deviate similarly from their references. If one is changing much more than the other, adjust our confidence accordingly.
        # token_constraint = layer_vals['rej_position_deviation'] - layer_vals['cho_position_deviation']
        # OR Only penalize when rejected deviates more (prevent sneaky gaming)
        alpha_token = 0.5
        loss = loss - alpha_token * token_constraint


    # Apply policy weights if requested
    if use_policy_weights:
        policy_weights = compute_policy_weights(pi_cho, pi_rej)
        vals['policy_weights'] = policy_weights.mean()
        vals['cho_log_policy_weights'] = torch.exp(pi_cho.log_policy_weights).mean()
        vals['rej_log_policy_weights'] = torch.exp(pi_rej.log_policy_weights).mean()   
        loss = loss * policy_weights.detach()

    vals = {k:v.mean() for k, v in vals.items()}
    info = dict(
        **vals,
    )

    return loss.mean(), info


@dataclass
class TopKLossConfig:
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

    inner_policy_weights: bool = False
    """Whether to compute policy weights for the inner DPO loss."""

    use_token_constraint: bool = False
    """Whether to use the token constraint to adjust the hidden ptheta. This is like in TDPO https://arxiv.org/abs/2404.11999"""
    # FIXME also add to outer DPO

    detach_ref: bool = False
    """Whether to detach the reference hidden states from the computation graph. This is useful to prevent the reference model from changing during training, which is less desired as it doesn't actually come up during generation"""

    # use_dpo_loss: bool = True

    def c(self, *args, **kwargs):
        return topk_loss(*args, **kwargs, **asdict(self))
