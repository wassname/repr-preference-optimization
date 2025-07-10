from jaxtyping import Float
from typing import Any, Callable, Dict, Optional, Literal
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict
from einops import rearrange
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

def topkl_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    # α: float = 1.0,
    eps: float = 1e-6,
    # β: float = 1,
    use_wpo: bool = False,
    # inner_policy_weights: bool = False,
    # align_method: str = 'para_signed',
    # norm_before_reduce: bool = True,
    filter_sinks: bool = False,
    trust_region: float = 2.0,
    # dpo_loss: str = "ipo",
    # p=2,
    # margin: float = 2,
    use_mallows: bool = False,
    # label_smoothing=0,
    # clamp_bottom: bool = False,
    detach_ref: bool = False,
    # use_token_constraint: bool = True,
    # token_con_alpha: float = 0.5,
    topk_n: int = 100
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
        hs_ref_cho_t = preproc_hs(ref_cho, k)
        hs_ref_rej_t = preproc_hs(ref_rej, k)
        pi_cho_mask = pi_cho.mask.clone()
        pi_rej_mask = pi_rej.mask.clone()
        ref_cho_mask = ref_cho.mask.clone()
        ref_rej_mask = ref_rej.mask.clone()
        if detach_ref:
            hs_pi_rej_t = hs_pi_rej_t.detach() # We want to change cho not rej

        if use_mallows:
            # weighting by mallows means we loose the last token, which has no labels and therefore no logits
            hs_pi_cho_t = compute_mallows_weights(hs_pi_cho_t[:, -1:], pi_cho.mallows_weights.unsqueeze(2))
            hs_pi_rej_t = compute_mallows_weights(hs_pi_rej_t[:, -1:], pi_rej.mallows_weights.unsqueeze(2))
            hs_ref_cho_t = compute_mallows_weights(hs_ref_cho_t[:, -1:], ref_cho.mallows_weights.unsqueeze(2))
            hs_ref_rej_t = compute_mallows_weights(hs_ref_rej_t[:, -1:], ref_rej.mallows_weights.unsqueeze(2))
            pi_cho_mask = pi_cho_mask[:, :-1]
            pi_rej_mask = pi_rej_mask[:, :-1]
            ref_cho_mask = ref_cho_mask[:, :-1]
            ref_rej_mask = ref_rej_mask[:, :-1]


        if use_wpo:
            # can be wpo, or mallows
            policy_weights = compute_policy_weights(pi_cho, pi_rej)
            hs_pi_cho_t = hs_pi_cho_t[:, :-1, :] * policy_weights.unsqueeze(2).detach()
            hs_pi_rej_t = hs_pi_rej_t[:, :-1, :] * policy_weights.unsqueeze(2).detach()
            hs_ref_cho_t = hs_ref_cho_t[:, :-1, :] * ref_cho.policy_weights.unsqueeze(2).detach()
            hs_ref_rej_t = hs_ref_rej_t[:, :-1, :] * ref_rej.policy_weights.unsqueeze(2).detach()

            # remove last token, which has no labels
            pi_cho_mask = pi_cho_mask[:, :-1]
            pi_rej_mask = pi_rej_mask[:, :-1]
            ref_cho_mask = ref_cho_mask[:, :-1]
            ref_rej_mask = ref_rej_mask[:, :-1]

        hs_pi_cho = reduce_tokens_w_attention(hs_pi_cho_t, pi_cho_mask, filter_sinks=filter_sinks)  # [batch, hidden_dim],
        hs_pi_rej = reduce_tokens_w_attention(hs_pi_rej_t, pi_rej_mask, filter_sinks=filter_sinks) 
        hs_ref_cho = reduce_tokens_w_attention(hs_ref_cho_t, ref_cho_mask, filter_sinks=filter_sinks)
        hs_ref_rej = reduce_tokens_w_attention(hs_ref_rej_t, ref_rej_mask, filter_sinks=filter_sinks)

        # Reference preference direction
        ref_pref = hs_ref_cho - hs_ref_rej
        pi_pref = hs_pi_cho - hs_pi_rej

        return dict(pi_pref=pi_pref,
                    ref_pref=ref_pref,
            )


    # compute losses per layer
    layer_vals = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(layer_vals.values())).keys()
    layer_vals = {k: torch.stack([v[k] for v in layer_vals.values()], 1) for k in ll_keys}

    # TODO add topk here, after norm
    # whats the shape? we stacked on dim=-1, so [batch, layers, hidden_dim]
    pi_pref = safe_norm(layer_vals['pi_pref'], dim=-1, eps=eps)
    ref_pref = safe_norm(layer_vals['ref_pref'], dim=-1, eps=eps)
    # movement = hs_pi_cho - hs_ref_cho
    pi_pref = rearrange(pi_pref, 'b l d -> b (d l)')  # [batch, hidden_dim, layers]
    ref_pref = rearrange(ref_pref, 'b l d -> b (d l)')  # [batch, hidden_dim, layers]

    # Find top-k dimensions where reference has strongest preference
    k = min(topk_n, ref_pref.shape[-1] // 4)
    topk_vals, topk_idx = torch.topk(ref_pref.abs(), k=k, dim=-1)
    # movement_topk = torch.gather(movement, -1, topk_idx)
    ref_pref_topk = torch.gather(ref_pref, -1, topk_idx)
    pi_pref_topk = torch.gather(pi_pref, -1, topk_idx)

    # Normalize reference direction
    # ref_norm = F.normalize(ref_pref_topk, dim=-1)
    pi_projection = (pi_pref_topk * F.normalize(ref_pref_topk, dim=-1)).sum(-1, keepdim=True)

    
    # Ratio: how much larger is policy preference vs reference
    # Values > 1 mean policy separates more than reference
    ref_magnitude = torch.norm(ref_pref_topk, dim=-1, keepdim=True) 
    preference_ratio = pi_projection / (ref_magnitude + 1e-8)
    
    # Trust region: we want ratio between [1.0, trust_region]
    # Below 1.0 is bad (policy separates less than reference)
    # Above trust_region risks incoherence
    lower_violation = F.relu(1.0 - preference_ratio)
    upper_violation = F.relu(preference_ratio - trust_region)
    
    # Loss encourages ratio = trust_region (maximum safe separation)
    target_loss = (trust_region - preference_ratio) ** 2
    
    loss = target_loss.mean() + lower_violation.mean() + 2 * upper_violation.mean()

    vals = {k: v.mean(-1) for k, v in layer_vals.items()}  # average over layers

    # loss = layer_vals['inner_loss']

    vals = {k:v.mean() for k, v in vals.items() if v is not None}  # reduce to scalar values
    info = dict(
        **vals,
    )
    # info['token_constraint'] = token_constraint.mean()

    return loss.mean(), info


@dataclass
class TopKLLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states apart along the preference vector, with some constraints, while also doing DPO on outpts
    """

    topk_n: int = 100
    """Number of top-k dimensions to consider for the loss"""

    filter_sinks: bool = False
    """Whether to filter attention sinks in the hidden states."""

    eps: float = 1.0e-4


    trust_region: float = 2.0
    """Trust region for the alignment loss, i.e. how much the hidden states can deviate from the reference before we penalize them."""


    use_mallows: bool = False
    """Whether to use Mallows weights"""


    detach_ref: bool = False
    """Whether to detach the reference hidden states from the computation graph. This is useful to prevent the reference model from changing during training, which is less desired as it doesn't actually come up during generation"""


    def c(self, *args, **kwargs):
        return topkl_loss(*args, **kwargs, **asdict(self))
