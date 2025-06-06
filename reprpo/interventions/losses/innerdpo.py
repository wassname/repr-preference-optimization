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
    align_method: str = 'direct_projection',
    norm_before_reduce: bool = True,
):
    """
    movement of hs along the hs pref vector.
    """

    def preproc_hs(o, k: str):
        """Preprocess hidden states: normalize then aggregate."""

        hs = o.hs[k]  # [batch, seq_len, hidden_dim], RAW ACTIVATIONS
        # Normalize to unit sphere FIRST (before aggregation)
        # This prevents token magnitude bias (e.g., attention sinks)
        hs = transforms.transforms[k](hs)
        # Aggregate over sequence using attention masks
        # hs = F.log_softmax(hs, dim=-1)  # [batch, seq_len, hidden_dim], LOG PROBABILITIES
        if norm_before_reduce:
            hs = F.normalize(hs, p=2, dim=-1)  # [batch, seq_len, hidden_dim], UNIT VECTORS. If we normalise before transforms, we get NaNs in the gradients
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
        par_comp_pi = torch.sum(pref_dir_pi * pref_dir_ref_unit, dim=-1, keepdim=True) * pref_dir_ref_unit
        orth_comp_pi = pref_dir_pi - par_comp_pi
        par_comp_ref = torch.sum(pref_dir_ref * pref_dir_ref_unit, dim=-1, keepdim=True) * pref_dir_ref_unit
        orth_comp_ref = pref_dir_ref - par_comp_ref


        def safe_log(x: Float[Tensor, "batch"], eps=eps):
            """Safe log function to avoid log(0) issues."""
            # return torch.log(x.clamp(min=eps))
            return torch.log(x+eps)

        # Magnitudes
        par_pi = torch.norm(par_comp_pi, p=1, dim=-1)
        ort_pi = torch.norm(orth_comp_pi, p=1, dim=-1)
        log_par = safe_signed_log(par_pi, eps=eps)
        log_ort = safe_signed_log(ort_pi, eps=eps)
        logodds_pi = log_par - log_ort
        par_ref = torch.norm(par_comp_ref, p=1, dim=-1)
        ort_ref = torch.norm(orth_comp_ref, p=1, dim=-1)
        log_par_ref = safe_signed_log(par_ref, eps=eps)
        log_ort_ref = safe_signed_log(ort_ref, eps=eps)
        logodds_ref = log_par_ref - log_ort_ref

        # # δ is a hyperparameter that controls the sensitivity of the tanh gate
        # δ = 0.1  # Hyperparameter for tanh gate, can be tuned
        # tot_mag = par_pi + ort_pi                     # how big was the movement?
        # mag_logit = torch.log(tot_mag + eps)   
        # gate = torch.clamp(torch.tanh(torch.norm(pref_dir_pi, p=2, dim=-1) / δ), min=0.1)

        # Make weights simialr to # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
        scores = torch.stack([log_par, log_ort], dim=-1)    # shape [B,2]
        log_hidden_probs = F.log_softmax(scores, dim=-1)   # [B,2]
        w_adj = torch.logsumexp(2 * log_hidden_probs, dim=-1)  # [B]
        pseudo_lp = log_hidden_probs[...,0] - w_adj     # “chosen” class
        hidden_weight = torch.clamp(torch.exp(pseudo_lp), max=1.0).detach()  # [B]

        match align_method:
            case 'abs':
                # Create a "preference logit" in hidden space
                hidden_ptheta = torch.abs(cho_score - rej_score) * β
            case 'cosine_similarity':
                # GIVES INCOHERENT OUTPUTS QUICKLY, this is a directional loss that is balanced with DPO
                # Compute preference directions

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
                # Project pi direction onto ref direction (signed distance)
                projection = (pref_dir_pi * pref_dir_ref_unit)  # [batch]
                proj_dist = torch.norm(projection, p=1, dim=-1)  # Magnitude of projection
                # normalise per layer?
                
                hidden_ptheta = β * proj_dist
                hidden_weight = torch.tensor(1.0, device=par_pi.device)  # No additional weight scaling
            case 'direct_projection_log':                
                # Project pi direction onto ref direction (signed distance)
                projection = (pref_dir_pi * pref_dir_ref_unit)  # [batch]
                proj_dist = torch.norm(projection, p=1, dim=-1)  # Magnitude of projection
                # normalise per layer?
                
                hidden_ptheta = β * torch.log1p(proj_dist+eps)  # Log of projection magnitude, scaled by β
                hidden_weight = torch.tensor(1.0, device=par_pi.device)  # No additional weight scaling
            case 'para_orth':
                # Preference for parallel over orthogonal, compared to base model
                hidden_ptheta = β * (logodds_pi - logodds_ref)
            case 'para_orth2':
                # DPO‐style log‐odds in log-domain (stable)
                odds = (par_pi - par_ref)  - (ort_pi-ort_ref)  # no ratios, no logs

                # Preference for parallel over orthogonal
                hidden_ptheta = β * odds
            case 'orth':                
                # Preference for not  orthogonal
                hidden_ptheta = - β * torch.log(ort_pi + eps)
                hidden_weight = torch.tensor(1.0, device=par_pi.device)  # No additional weight scaling
            case 'para':
                # Preference for parallel 
                hidden_ptheta = β * torch.log(par_pi + eps)
                hidden_weight = torch.tensor(1.0, device=par_pi.device)  # No additional weight scaling
            case 'angle_mag':
                # ah this just ends up encouraging a low magnitude for a low loss
                # Standard alignment
                alignment = F.cosine_similarity(pref_dir_pi, pref_dir_ref, dim=-1)
                prob = torch.abs(alignment)
                prob = (prob - eps).clamp(min=eps)
                log_odds = torch.atanh(prob)
                
                # Scale by reference magnitude (weak preferences get less loss)
                ref_magnitude = torch.norm(pref_dir_ref, dim=-1)
                magnitude_weight = torch.clamp(ref_magnitude, 0.1, 2.0)  # Reasonable range
                
                hidden_ptheta = β * log_odds / magnitude_weight  # Magnitude-weighted log-odds
        
        # Apply DPO-style loss
        loss_hidden_dpo = -F.logsigmoid(hidden_ptheta)
        # if use_policy_weights: # I'm not sure this is helping
        #     loss_hidden_dpo = loss_hidden_dpo * hidden_weight
        
        return dict(loss_hidden_dpo=loss_hidden_dpo, hidden_weight=hidden_weight, hidden_ptheta=hidden_ptheta)


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


    align_method: str = 'para'
    """Method to compute alignment between chosen and rejected hidden states."""

    norm_before_reduce: bool = True
    """Whether to normalize hidden states before reducing them to a single vector."""
   

    def c(self, *args, **kwargs):
        return innerdpo_loss(*args, **kwargs, **asdict(self))
