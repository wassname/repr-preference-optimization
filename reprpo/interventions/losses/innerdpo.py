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

def innerdpo_loss(
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
    use_policy_weights: bool = False,
    align_method: str = 'direct_projection',
):
    """
    movement of hs along the hs pref vector.
    """

    def preproc_hs(o, k: str):
        """Preprocess hidden states: normalize then aggregate."""

        hs = o.hs[k]  # [batch, seq_len, hidden_dim], RAW ACTIVATIONS
        # Normalize to unit sphere FIRST (before aggregation)
        # This prevents token magnitude bias (e.g., attention sinks)
        hs = F.normalize(hs, p=2, dim=-1)  # [batch, seq_len, hidden_dim], UNIT VECTORS
        # Aggregate over sequence using attention masks
        # hs = F.log_softmax(hs, dim=-1)  # [batch, seq_len, hidden_dim], LOG PROBABILITIES
        hs = transforms.transforms[k](hs)
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


        match align_method:
            case 'abs':
                # Create a "preference logit" in hidden space
                hidden_ptheta = torch.abs(cho_score - rej_score) * β
            case 'cosine_similarity':
                # GIVES INCOHERENT OUTPUTS QUICKLY, this is a directional loss that is balanced with DPO
                # Compute preference directions
                pref_dir_ref = hs_ref_cho - hs_ref_rej
                pref_dir_pi = hs_pi_cho - hs_pi_rej

                # Alignment score [-1, 1]
                alignment = F.cosine_similarity(pref_dir_pi, pref_dir_ref, dim=-1)
                
                # Convert to log-odds (proper log ratio)
                # Map [-1,1] to [0,1] then to log-odds
                prob = torch.abs(alignment) # (alignment + 1) / 2  # [-1,1] -> [0,1]
                # prob = torch.clamp(prob, min=eps, max=1 - eps)  # Avoid log(0) issues
                prob = (prob-eps).clamp(min=eps)
                log_odds = torch.atanh(prob)  # (0,1] -> [-∞,∞]

                #z This is now a proper log ratio!
                hidden_ptheta = β * log_odds

                # OR
                # hidden_ptheta = β * torch.logit(prob/(prob+1))  # [0, ∞)
            case 'log_ratio':
                # log ratio
                hidden_ptheta = (torch.log(cho_score.abs()) - torch.log(rej_score.abs())) * β  # Difference of log similarities
            case 'direct_projection':
                # Compute preference directions
                pref_dir_ref = hs_ref_cho - hs_ref_rej  # [batch, hidden_dim]
                pref_dir_pi = hs_pi_cho - hs_pi_rej     # [batch, hidden_dim]
                
                # Normalize reference direction to unit vector
                pref_dir_ref_unit = F.normalize(pref_dir_ref, p=2, dim=-1)
                
                # Project pi direction onto ref direction (signed distance)
                projection = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1)  # [batch]
                
                hidden_ptheta = β * projection
            case 'parrel_orthogonal':
                pref_dir_ref = hs_ref_cho - hs_ref_rej
                pref_dir_pi = hs_pi_cho - hs_pi_rej
                
                # Decompose pi into parallel and orthogonal components
                ref_unit = F.normalize(pref_dir_ref, p=2, dim=-1)
                parallel_component = torch.sum(pref_dir_pi * ref_unit, dim=-1, keepdim=True) * ref_unit
                orthogonal_component = pref_dir_pi - parallel_component
                
                # Magnitudes
                parallel_mag = torch.norm(parallel_component, p=2, dim=-1)
                orthogonal_mag = torch.norm(orthogonal_component, p=2, dim=-1)
                
                # Preference for parallel over orthogonal
                hidden_ptheta = β * (torch.log(parallel_mag + eps) - torch.log(orthogonal_mag + eps))
            case 'angle_mag':
                pref_dir_ref = hs_ref_cho - hs_ref_rej
                pref_dir_pi = hs_pi_cho - hs_pi_rej
                
                # Standard alignment
                alignment = F.cosine_similarity(pref_dir_pi, pref_dir_ref, dim=-1)
                prob = torch.abs(alignment)
                prob = (prob - eps).clamp(min=eps)
                log_odds = torch.atanh(prob)
                
                # Scale by reference magnitude (weak preferences get less loss)
                ref_magnitude = torch.norm(pref_dir_ref, dim=-1)
                magnitude_weight = torch.clamp(ref_magnitude, 0.1, 2.0)  # Reasonable range
                
                hidden_ptheta = β * magnitude_weight * log_odds  # Magnitude-weighted log-odds
        
        # Apply DPO-style loss
        loss_hidden_dpo = -F.logsigmoid(hidden_ptheta)
        
        return dict(loss_hidden_dpo=loss_hidden_dpo)


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
    if use_policy_weights:
        policy_weights = torch.clamp(
            torch.exp(pi_cho.log_policy_weights + pi_rej.log_policy_weights),
            max=1
        )
        loss = loss * policy_weights.mean()
    


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
    - keep text at least as coherent (relu(mode/base), (nll_loss)
    - keep the chosen answer at least prefered (relu(rej-cho) dpo_loss
    - punish movement orthogonal to the preference vector: by distance * β
    - punish movement orthogonal to the preference vector: by angle * β
    """

    α: float = 0.1
    """balance between reroute and retain loss."""

    eps: float = 1.0e-5

    β: float = 1.
    """factor to punish orthogonal movement"""

    use_policy_weights: bool = True
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""


    align_method: str = 'direct_projection'
    """Method to compute alignment between chosen and rejected hidden states."""
   

    def c(self, *args, **kwargs):
        return innerdpo_loss(*args, **kwargs, **asdict(self))
