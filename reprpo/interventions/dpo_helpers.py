import torch
import torch.nn.functional as F


def compute_logprobs(logits, labels, selection_mask=None, type="ipo"):
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
    labels = labels[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        if type == "dpo":
            output["label_logp"] = selected_log_probs.sum(
                -1
            )  # sum over logprobs, total prob of whole completion
        elif type == "ipo":
            # Calculate the average log probability excluding padding tokens
            # This averages over the tokens, so the shape is (batch_size, num_tokens)
            output["label_logp"] = selected_log_probs.sum(-1) / mask.sum(-1)

    else:
        output["label_logp"] = selected_log_probs.mean(-1)

        selected_log_probs[~selection_mask] = 0
    

    # return a dict and also compute [WPO](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L1240)
    with torch.no_grad():
        # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
        weights_adjustment_factor = torch.logsumexp(2 * log_probs, dim=-1)  # same as sum(probs**2) in log space
        per_token_logps_adjusted = selected_log_probs - weights_adjustment_factor
        weights = (per_token_logps_adjusted * selection_mask).sum(-1) / selection_mask.sum(-1)
        output["policy_weights"] =  torch.clamp(torch.exp(weights), max=1)

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
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
):
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios
    return logits
