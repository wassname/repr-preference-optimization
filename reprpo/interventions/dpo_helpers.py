import torch
import torch.nn.functional as F
from loguru import logger
from jaxtyping import Float, Int
from typing import Dict, Optional
from torch import Tensor


def compute_logprobs(logits, input_ids, selection_mask=None, calc_wpo=False, calc_mallows=False):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.


    see https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L1240
    """

    output = {}

    # Labels are the inputs shifted by one
    labels = input_ids[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    output["label_logp"] = selected_log_probs

    # if selection_mask is not None:
    #     mask = selection_mask[:, :-1].clone()
    #     # Apply the mask to filter out padding tokens
    #     # selected_log_probs[~mask] = 0
    #     selected_log_probs = selected_log_probs * mask

    #     if logp_agg_type == "dpo":
    #         output["label_logp"] = selected_log_probs.sum(
    #             -1
    #         )  # sum over logprobs, total prob of whole completion
    #     elif logp_agg_type == "ipo":
    #         # Calculate the average log probability excluding padding tokens
    #         # This averages over the tokens, so the shape is (batch_size, num_tokens)
    #         output["label_logp"] = selected_log_probs.sum(-1) / mask.sum(-1)
    #         assert all(mask.sum(-1) > 0), "Mask should not be all zeros, check your input data."

    # else:
    #     logger.warning(
    #         "No selection mask provided, using all tokens for log probability calculation."
    #     )
    #     raise ValueError(
    #         "Selection mask is required for DPO loss calculation. Please provide a valid selection mask."
    #     )
    #     output["label_logp"] = selected_log_probs.mean(-1)
    
    
    # return a dict and also compute 
    # output["log_policy_weights"] = torch.zeros_like(output["label_logp"])
    if calc_wpo and calc_mallows:
        raise ValueError("Cannot use both WPO and Mallows weights at the same time. Choose one.")
    if calc_wpo:
        # [WPO](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L1240)
        with torch.no_grad():
            # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
            weights_adjustment_factor = torch.logsumexp(2 * log_probs, dim=-1)  # same as sum(probs**2) in log space
            per_token_logps_adjusted = selected_log_probs - weights_adjustment_factor
            # weights = (per_token_logps_adjusted * mask).sum(-1) / mask.sum(-1)
            weights = per_token_logps_adjusted 
            output["log_policy_weights"] =  weights.detach()
    
    if calc_mallows:
        # https://github.com/CapitalOne-Research/RainbowPO/blob/main/trl/trainer/dpo_trainer.py#L1347
        with torch.no_grad():
            dispersion_mean = 1.0
            log_vocab_size = torch.log(torch.tensor(logits.shape[-1]))

            # Per-token entropy: [batch, seq_len]
            token_entropy = -(log_probs.exp() * log_probs).sum(-1)

            # Normalize and convert to neg-log dispersion
            token_entropy_normalized = token_entropy / log_vocab_size
            neg_log_dispersion = -dispersion_mean * torch.log(token_entropy_normalized + 1e-8)

            # Apply loss mask to zero out padded positions
            neg_log_dispersion = neg_log_dispersion * selection_mask[:, :-1]

            # output["mallows_weights"] = (batch_entropy * mask).sum(axis = -1) / (mask.sum(-1) * log_vocab_size)
            output["mallows_weights"] = token_entropy.detach()

    return output


def calc_mallows_weights(
    logits: Float[Tensor, "b t v"],
    log_probs: Float[Tensor, "b t v"],
    selection_mask: Float[Tensor, "b t"],
    dispersion_mean = 1.0
) -> Dict[str, Tensor]:
    """
    Token-level entropy weighting for DPO hidden state optimization.
    
    This implementation adapts the MallowsPO dispersion concept from sequence-level to token-level granularity.
    
    Background:
    -----------
    MallowsPO (Chen et al., 2024) introduces a dispersion index φ(x) to weight preference pairs based on 
    the uncertainty/disagreement in human preferences. The key insight is that some prompts have clear 
    "correct" answers (low entropy) while others are subjective (high entropy).
    
    Rainbow (2024) implements this by calculating the average entropy of chosen/rejected responses using
    the reference model, then applying a scaling factor: -dispersion_mean * log(entropy).
    
    This Implementation:
    -------------------
    We extend the concept to token-level weighting within sequences. Instead of weighting entire 
    preference pairs, we weight individual tokens based on their prediction entropy. The hypothesis 
    is that tokens with lower entropy (higher certainty) should contribute more to learning, while 
    high-entropy tokens may represent inherent ambiguity or noise.
    
    Key differences from original MallowsPO:
    1. Token-level vs sequence-level: We weight each token individually rather than the entire sequence
    2. Policy + Reference models: We use entropy from both models (original uses only reference)
    3. Hidden state weighting: Applied to hidden representations for inner optimization
    4. Granular learning: Enables the model to focus on "easier" tokens within each sequence
    
    Mathematical formulation:
    For each token position i:
        entropy_i = -Σ p_i * log(p_i)  where p_i are the softmax probabilities
        weight_i = exp(-entropy_i / temperature)  # Higher weight for lower entropy

    """
    log_vocab_size = torch.log(torch.tensor(logits.shape[-1]))

    # Per-token entropy: [batch, seq_len]
    token_entropy = -(log_probs.exp() * log_probs).sum(-1)

    # Normalize and convert to neg-log dispersion
    token_entropy_normalized = token_entropy / log_vocab_size
    neg_log_dispersion = -dispersion_mean * torch.log(token_entropy_normalized + 1e-8)

    # Apply loss mask to zero out padded positions
    neg_log_dispersion = neg_log_dispersion * selection_mask

    return token_entropy.detach()

def compute_mallows_weights(hs, weight, mask=None) -> Float[Tensor, "b t"]:
    """weight hs by mallow weights"""
    o =  hs * weight
    
    if mask is not None:
        o = (o * mask.unsqueeze(-1))/ (mask.sum(-1, keepdim=True).unsqueeze(-1) + 1e-6)
    return o


def compute_policy_weights(pi_cho, pi_rej, T=3) -> Float[Tensor, "b t"]:
    # Here we deviate from the WPO paper for stability
    policy_weights = torch.exp(pi_cho.log_policy_weights + pi_rej.log_policy_weights)
    # OR
    policy_weights = torch.sigmoid((pi_cho.log_policy_weights + pi_rej.log_policy_weights)/T)

    # balance them
    policy_weights = policy_weights /(policy_weights.mean() + 1e-6)
    return policy_weights

def cross_entropy_loss(logits, labels, attn):
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    logits2 = logits.view(-1, logits.shape[-1])
    labels2 = labels.view(-1)
    # Enable model parallelism
    labels2 = labels2.to(logits.device)
    loss = loss_fct(logits2, labels2).view_as(labels) * attn.detach()
    return loss


def compute_ptheta(
    pi_cho_logp,
    pi_rej_logp,
    ref_cho_logp,
    ref_rej_logp,
):
    pi_logratios = pi_cho_logp - pi_rej_logp
    ref_logratios = ref_cho_logp - ref_rej_logp
    logits = pi_logratios - ref_logratios.detach()
    return logits

