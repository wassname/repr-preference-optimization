import torch
import torch.nn.functional as F
from loguru import logger
from jaxtyping import Float, Int
from typing import Dict, Optional
from torch import Tensor


def compute_logprobs(logits, input_ids, selection_mask=None, logp_agg_type="ipo", calc_wpo=False, calc_mallows=False):
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


    if selection_mask is not None:
        mask = selection_mask[:, :-1].clone()
        # Apply the mask to filter out padding tokens
        # selected_log_probs[~mask] = 0
        selected_log_probs = selected_log_probs * mask

        if logp_agg_type == "dpo":
            output["label_logp"] = selected_log_probs.sum(
                -1
            )  # sum over logprobs, total prob of whole completion
        elif logp_agg_type == "ipo":
            # Calculate the average log probability excluding padding tokens
            # This averages over the tokens, so the shape is (batch_size, num_tokens)
            output["label_logp"] = selected_log_probs.sum(-1) / mask.sum(-1)
            assert all(mask.sum(-1) > 0), "Mask should not be all zeros, check your input data."

    else:
        logger.warning(
            "No selection mask provided, using all tokens for log probability calculation."
        )
        raise ValueError(
            "Selection mask is required for DPO loss calculation. Please provide a valid selection mask."
        )
        output["label_logp"] = selected_log_probs.mean(-1)
    
    
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
            log_vocab_size = torch.log(torch.tensor(logits.shape[-1]))
            batch_entropy = -(log_probs.exp() * log_probs).sum(-1)
            # output["mallows_weights"] = (batch_entropy * mask).sum(axis = -1) / (mask.sum(-1) * log_vocab_size)
            output["mallows_weights"] = batch_entropy.detach() / log_vocab_size

    return output


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


def compute_policy_weights(pi_cho, pi_rej, T=3) -> Float[Tensor, "b t"]:
    # Here we deviate from the WPO paper for stability
    policy_weights = torch.exp(pi_cho.log_policy_weights + pi_rej.log_policy_weights)
    # OR
    policy_weights = torch.sigmoid((pi_cho.log_policy_weights + pi_rej.log_policy_weights)/T)

    # balance them
    policy_weights = policy_weights /(policy_weights.mean() + 1e-6)
    return policy_weights

def compute_mallows_weights(ref_cho, ref_rej, dispersion_mean = 0.29, reduce=False) -> Float[Tensor, "b t"]:
    # This approach upweights tokens with higher entropy (more uncertainty), 
    # from https://github.com/CapitalOne-Research/RainbowPO/blob/main/trl/trainer/dpo_trainer.py#L1276
    ref_rej_mallows_weights = ref_rej.mallows_weights
    ref_cho_mallows_weights = ref_cho.mallows_weights
    if reduce:
        ref_rej_mallows_weights = (ref_rej_mallows_weights * ref_rej.mask[:, :-1]).sum(dim=1) / ref_rej.mask[:, :-1].sum(dim=1)
        ref_cho_mallows_weights = (ref_cho_mallows_weights * ref_cho.mask[:, :-1]).sum(dim=1) / ref_cho.mask[:, :-1].sum(dim=1)
    # Here we deviate from the WPO paper for stability
    # TODO dispersion_mean should ideally be a moving average?
    reference_entropy = (ref_cho_mallows_weights + ref_rej_mallows_weights)/2
    neg_log_dispersion = - dispersion_mean * torch.log(reference_entropy)
    if reference_entropy.sum() == 0:
        # Reference entropy not provided, wallow weights will be zero.
        neg_log_dispersion = None
    return neg_log_dispersion
