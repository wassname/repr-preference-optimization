from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from ..dpo_helpers import cross_entropy_loss, compute_ptheta
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention

def safe_signed_log(x: Tensor, eps: float = 1e-12):
    # preserve the sign, only clamp the magnitude
    sign = x.sign()
    mag  = x.abs().clamp(min=eps)
    return sign * torch.log(mag)

def innerpo_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps=1e-4,
    β=1,
    use_orth_loss=True,
    use_dpo_loss=True,
    use_proj_loss=False,
    use_proj_abs_loss=True,
    use_logsigmoid=True,
    use_policy_weights: bool = False,
):
    """
    movement of hs along the hs pref vector.
    """

    # # TODO norm or softmax each layer before transform?
    # for k in pi_cho.hs.keys():
    #     pi_cho.hs[k] = F.normalize(pi_cho.hs[k], p=2, dim=-1)
    #     pi_rej.hs[k] = F.normalize(pi_rej.hs[k], p=2, dim=-1)
    #     ref_cho.hs[k] = F.normalize(ref_cho.hs[k], p=2, dim=-1)
    #     ref_rej.hs[k] = F.normalize(ref_rej.hs[k], p=2, dim=-1)

    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k: str):
        """Preprocess hidden states: normalize then aggregate."""
        hs = o.hs[k]  # [batch, seq_len, hidden_dim], RAW ACTIVATIONS
        # Normalize to unit sphere FIRST (before aggregation)
        # This prevents token magnitude bias (e.g., attention sinks)
        hs = F.normalize(hs, p=2, dim=-1)  # [batch, seq_len, hidden_dim], UNIT VECTORS
        # Aggregate over sequence using attention masks
        # hs = F.log_softmax(hs, dim=-1)  # [batch, seq_len, hidden_dim], LOG PROBABILITIES
        hs = reduce_tokens_w_attention(hs, o.mask)  # [batch, hidden_dim], AVERAGED UNIT VECTORS
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k):
        # Get raw hidden states
        hs_pi_cho = reduce_tokens_w_attention(pi_cho.hs[k], pi_cho.mask)
        hs_pi_rej = reduce_tokens_w_attention(pi_rej.hs[k], pi_rej.mask)
        hs_ref_cho = reduce_tokens_w_attention(ref_cho.hs[k], ref_cho.mask)
        hs_ref_rej = reduce_tokens_w_attention(ref_rej.hs[k], ref_rej.mask)
        
        # Compute similarity scores (like logits in DPO)
        cho_score = F.cosine_similarity(hs_pi_cho, hs_ref_cho, dim=-1)
        rej_score = F.cosine_similarity(hs_pi_rej, hs_ref_rej, dim=-1)
        
        # Create a "preference logit" in hidden space
        hidden_ptheta = (cho_score - rej_score) * β
        
        # Apply DPO-style loss
        loss_hidden_dpo = -F.logsigmoid(hidden_ptheta)
        
        return dict(loss_hidden_dpo=loss_hidden_dpo)


    # def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
    #     """Compute losses for a single layer."""
        
    #     # Get aggregated, normalized hidden states
    #     # Domain: unit vectors averaged, so ||hs|| ≤ 1
    #     hs_pi_cho = preproc_hs(pi_cho, k)   # [b, d], averaged unit vectors
    #     hs_pi_rej = preproc_hs(pi_rej, k)   # [b, d], averaged unit vectors
    #     hs_ref_cho = preproc_hs(ref_cho, k) # [b, d], averaged unit vectors
    #     hs_ref_rej = preproc_hs(ref_rej, k) # [b, d], averaged unit vectors

    #     # Compute preference directions as vector differences
    #     # Domain: VECTOR SPACE, each component ∈ [-2, 2] since we subtract unit vectors
    #     pref_dir_ref = hs_ref_cho - hs_ref_rej  # [b, d], reference preference direction
    #     pref_dir_pi = hs_pi_cho - hs_pi_rej     # [b, d], model preference direction
        
    #     # How much did model move relative to reference?
    #     # Domain: VECTOR SPACE, each component ∈ [-4, 4]
    #     delta = pref_dir_pi - pref_dir_ref      # [b, d], change in preference
        
    #     # Get reference magnitude and unit direction
    #     # Domain: SCALAR DISTANCE ∈ [0, 2√d], clamped at eps
    #     ref_mag = pref_dir_ref.norm(dim=-1, keepdim=True).clamp(min=eps, max=10)  # [b, 1]
    #     # Domain: UNIT VECTOR
    #     ref_unit = pref_dir_ref / ref_mag      # [b, d], normalized reference direction

    #     # Project delta onto reference direction (signed distance)
    #     # Domain: SIGNED SCALAR DISTANCE ∈ [-4√d, 4√d]
    #     signed_proj = (delta * ref_unit).sum(dim=-1)  # [b], positive = moved in ref direction
        
    #     # Orthogonal component magnitude (Pythagorean theorem)
    #     # Domain: SCALAR DISTANCE ∈ [0, 4√d]
    #     orth_magnitude = torch.sqrt((delta**2).sum(dim=-1) - signed_proj**2 + eps)  # [b]
        
    #     # Convert to ratios by normalizing with reference magnitude
    #     # This makes quantities dimensionless and comparable across examples
        
    #     if use_proj_abs_loss:
    #         # For bidirectional training (e.g., reducing toxicity)
    #         # Domain: RATIO ∈ [0, ∞), 0=no movement, 1=same as ref, >1=more than ref
    #         proj_ratio = torch.abs(signed_proj) / ref_mag.squeeze(-1)  # [b]
    #     else:
    #         # For unidirectional training
    #         # Domain: SIGNED RATIO ∈ (-∞, ∞), negative=opposite direction
    #         proj_ratio = signed_proj / ref_mag.squeeze(-1)  # [b]
        
    #     # Orthogonal ratio (always want to minimize)
    #     # Domain: RATIO ∈ [0, ∞), 0=no orthogonal movement, 1=same as ref magnitude
    #     orth_ratio = orth_magnitude / ref_mag.squeeze(-1)  # [b]
        
    #     # Convert to log domain for logsigmoid compatibility
    #     # logsigmoid expects logits where positive=good, negative=bad        
    #     if use_proj_abs_loss:
    #         # Absolute projection: ratio ∈ [0, ∞)
    #         # log(ratio): 0→-∞ (bad), 1→0 (neutral), ∞→∞ (good)
    #         proj_logits = torch.log(proj_ratio + eps) * β  # [b], LOG DOMAIN
    #     else:
    #         # Signed projection
    #         proj_logits = safe_signed_log(proj_ratio) * β  # [b], LOG DOMAIN
        
    #     # Orthogonal: we want LOW ratios
    #     # So negative log: 0→∞ (good), 1→0 (neutral), ∞→-∞ (bad)
    #     orth_logits = -torch.log(orth_ratio + eps) * β  # [b], LOG DOMAIN
        
    #     # Convert to losses using logsigmoid
    #     # -logsigmoid(x) is high when x is negative, low when x is positive
    #     loss_proj = -F.logsigmoid(proj_logits)  # [b], want positive proj_logits
    #     loss_orth = -F.logsigmoid(orth_logits)  # [b], want positive orth_logits (low orth_ratio)
        
    #     return dict(
    #         loss_proj=loss_proj,
    #         loss_orth=loss_orth,
    #         proj_ratio=proj_ratio,
    #         orth_ratio=orth_ratio,
    #         # For debugging
    #         signed_proj=signed_proj,
    #         ref_mag=ref_mag.squeeze(-1),
    #     )

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
    loss_dpo = -F.logsigmoid(dpo_ptheta)


    loss = loss_dpo + α * ll['loss_hidden_dpo']
    # OR


    # # Combine losses
    # losses = []
    # if use_proj_loss:
    #     losses.append(ll['loss_proj'].mean())
    # if use_orth_loss:
    #     losses.append(ll['loss_orth'].mean())
    # if use_dpo_loss:
    #     losses.append(loss_dpo)
    # loss = sum(losses) / len(losses) if losses else 0.0
    
    # Apply policy weights if requested
    if use_policy_weights:
        policy_weights = torch.clamp(
            torch.exp(pi_cho['log_policy_weights'] + pi_rej['log_policy_weights']),
            max=1
        )
        loss = loss * policy_weights.detach()
        ll['policy_weights'] = policy_weights.mean()
    


    ll = {k:v.mean() for k, v in ll.items()}
    info = dict(
        # loss_orth_term=loss_orth_term.mean(),
        # loss_proj_term=loss_proj_term.mean(),
        # loss_dpo_term=loss_dpo_prob_term.mean(),
        # dpo_prob=dpo_prob.mean(),
        loss_dpo=loss_dpo.mean(),
        dpo_ptheta=dpo_ptheta.mean(),
        # loss_logsigmoid_dpo=loss_logsigmoid_dpo.mean(),
        **ll,
    )

    return loss.mean(), info


@dataclass
class InnerPOLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states apart along the preference vector, with some constraints, while also doing DPO on outpts
    - keep text at least as coherent (relu(mode/base), (nll_loss)
    - keep the chosen answer at least prefered (relu(rej-cho) dpo_loss
    - punish movement orthogonal to the preference vector: by distance * β
    - punish movement orthogonal to the preference vector: by angle * β
    """

    α: float = 0.1
    """balance between reroute and retain loss."""

    eps: float = 1.0e-6

    β: float = 1.
    """factor to punish orthogonal movement"""

    use_dpo_loss: bool = True
    """punish model if rejected completion is more likely than the chosen"""

    use_orth_loss: bool = True
    """punish movement orthogonal to the preference vector: by distance"""

    use_proj_loss: bool = False
    """encourage chosen to be more in the pref dir than rejected"""

    use_proj_abs_loss: bool = True
    """use absolute value of the projection loss, otherwise use relative"""

    use_logsigmoid: bool = True

    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""
   

    def c(self, *args, **kwargs):
        return innerpo_loss(*args, **kwargs, **asdict(self))
