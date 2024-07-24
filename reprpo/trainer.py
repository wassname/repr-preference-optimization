from dataclasses import dataclass
from einops import rearrange
from trl import DPOConfig, DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import gc
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float

# def norm(a: Float[Tensor, 'b l t h']) ->  Float[Tensor, 'b l t h']:
#     """normalise the hidden states."""
#     im = rearrange(a, 'b l t h -> b t (l h)')
#     # print(torch.norm(im, dim=-1, keepdim=True, p=1)+1)
#     im = im / (torch.norm(np.abs(im), dim=-1, keepdim=True, p=2)+1)
#     im = rearrange(im, 'b t (l h) -> b l t h', l=a.shape[1], h=a.shape[-1])
#     return im


def norm_t_h(a):
    eps = 1e-7
    im = rearrange(a, 'b l t h -> (b l) t h')
    im = im /  (torch.abs(im).max(0, keepdim=True).values+eps)
    im = rearrange(im, '(b l) t h -> b l t h', b=a.shape[0], l=a.shape[1])
    return im

def norm_h(a):
    eps = 1e-7
    im = rearrange(a, 'b l t h -> (b l t) h')
    im = im /  (torch.abs(im).max(0, keepdim=True).values+eps)
    im = rearrange(im, '(b l t) h -> b l t h', b=a.shape[0], l=a.shape[1], t=a.shape[2])
    return im


def mean_with_attention(x: Float[Tensor, 'b l t h'], attn_mask: Float[Tensor, 'b t']) -> Float[Tensor, 'b l h']:
    """mean of x, weighted by the attention mask."""
    attn_mask = attn_mask[:, None, :, None].repeat(1, x.shape[1], 1, 1)
    return (x * attn_mask).sum(2, keepdim=True) / attn_mask.sum(2, keepdim=True)

def symlog(x, eps=1e-6):
    return torch.sign(x) * torch.log(torch.abs(x)+eps)


def symlog_loss(x, y, eps=1e-6):
    # maybe a simple norm or x and y the same here? just to scale the outputs
    e = symlog(x)-symlog(y)
    # e = norm_h(e)
    return e**2

def sum_squared_error(input, target):
    # from https://github.com/GraySwanAI/circuit-breakers/blob/main/src/lorra_circuit_breaker.py
    return torch.norm(input - target, dim=-1, p=2, dtype=torch.float, keepdim=True)#.nanmean()

def combined_loss(x, y, alpha=0.5):
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    direction_loss = 1 - cos_sim
    magnitude_loss = F.l1_loss(torch.norm(x, dim=-1), torch.norm(y, dim=-1))
    return alpha * direction_loss + (1-alpha) * magnitude_loss

def cka_inspired_similarity(x, y):
    """This is inspired by CKA, which has been used to compare neural network representations. It's invariant to orthogonal transformations and isotropic scaling, which can be beneficial for comparing hidden states."""

    # treat all tokens as the same
    x = rearrange(x, "b l t h -> b l (t h)")
    y = rearrange(y, "b l t h -> b l (t h)")

    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    dot_product = torch.sum(x_centered * y_centered, dim=-1)
    norm_x = torch.norm(x_centered, dim=-1)
    norm_y = torch.norm(y_centered, dim=-1)
    return dot_product / (norm_x * norm_y + 1e-8)


def norm_smooth_l1_loss(x, y):
    """error between the norms of two tensors."""
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    norm_x = torch.norm(x_centered, dim=-1)
    norm_y = torch.norm(y_centered, dim=-1)
    return F.smooth_l1_loss(norm_x, norm_y)

def top_k_mse(x, y, k=0.005):
    k = int(k * x.shape[-1])+1
    diff = (x - y)**2
    # get the top k differences along dim
    top_k_diff = torch.topk(diff, k, dim=-1).values
    return torch.mean(top_k_diff, dim=-1)

def top_k_mse_all(x, y, k=0.005):
    x = rearrange(x, "b l t h -> b (l t h)")
    y = rearrange(y, "b l t h -> b (l t h)")
    k = int(k * x.shape[-1])+1
    diff = (x - y)**2
    # get the top k differences along dim
    top_k_diff = torch.topk(diff, k, dim=-1).values
    return torch.mean(top_k_diff, dim=-1)


def coeffecient(t, T):
    t = t % T
    c = t / (2 * T)
    c_retain, c_rr = c, (1 - c)
    return c_retain, c_rr


@dataclass
class ReprPOConfig(DPOConfig):
    collection_layers: tuple = (10, 20, 26)
    alpha: int = 1
    print_every: int = 10


def collect_hs(hs):
    """The residual stream or hs of the diff of the hs."""
    hs = rearrange(list(hs), "l b t h -> l b t h")
    return rearrange(hs, "l b t h -> b l t h")


def wmean(x, w, dim=0):
    """weighted mean per neuron over batch."""
    # assert w.sum(-1)==1
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return (x * w).sum(dim, keepdim=True) / w.sum(dim, keepdim=True)


class ReprPOTrainer(DPOTrainer):
    """modified to optimise representations, that is hidden states not outputs."""

    def __init__(self, args: Optional[ReprPOConfig] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.collection_layers = args.collection_layers
        self.alpha = args.alpha
        self.loss_type = 'ipo'

        self.num_training_steps = self.args.max_steps
        if self.num_training_steps==-1:
            self.num_training_steps = self.args.num_train_epochs * len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
        print('TODO QC: self.num_training_steps', self.num_training_steps)

    def get_training_progress(self):
        # in the paper they claim it's schedule but they end up making it loop every 300 steps, but then use 1500 steps for the loss
        return self.state.global_step / self.num_training_steps
    
    def get_coeff(self):
        c = self.get_training_progress() % 1
        c /= 2
        c_retain, c_rr = c, (1 - c)
        return c_retain, c_rr

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        # so this multiplies the probs and makes it quite small, in the log domain that's ok, it represents the log probs of the whole string
        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )
        outs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            **model_kwargs,
        )
        all_logits = outs.logits
        hs = collect_hs(outs.hidden_states)[:, self.collection_layers]
        del outs
        gc.collect()

        # if self.state.global_step % self.args.print_every == 0:
        #     # print the last 5 tokens, for the first in batch
        #     s = model.tokenizer.batch_decode(
        #         all_logits.detach().cpu().softmax(-1).argmax(-1)[0, :5:]
        #     )
        #     print(
        #         f'last_N_tok {model.active_adapter if model.get_model_status().enabled else ""} {s}'
        #     )

        # multiply by attention mask
        attn_mask = concatenated_batch["concatenated_attention_mask"]


        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        chosen_logps_avg = all_logps[:len_chosen] / size_completion[:len_chosen]

        # So we want sum of logprobs or mean of logprobs? Like IPO we will use the log prob per token, https://github.com/eric-mitchell/direct-preference-optimization/issues/40
        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_hs = hs[:len_chosen]
        rejected_hs = hs[len_chosen:]

        chosen_attn_mask = attn_mask[:len_chosen]
        rejected_attn_mask = attn_mask[len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_logps_avg,
            chosen_hs,
            rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model.eval()
        # with model.disable_adapter():
        with torch.no_grad():
            with self.null_ref_context():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                    reference_chosen_hs,
                    _,
                    _,
                    _
                ) = self.concatenated_forward(self.model, batch)
        reference_chosen_hs = reference_chosen_hs.detach()
        reference_chosen_logps = reference_chosen_logps.detach()
        reference_rejected_logps = reference_rejected_logps.detach()

        model.train()
        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
            policy_chosen_logps_avg,
            policy_chosen_hs,
            policy_rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask
        ) = self.concatenated_forward(model, batch)

        loss, loss_info = self.reprpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_hs,
            policy_rejected_hs,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_chosen_hs,
            chosen_attn_mask,
            rejected_attn_mask
        )
        # losses, chosen_rewards, rejected_rewards, loss_retain, loss_rr = loss_info
        # chosen_rewards, rejected_rewards = (
        #     loss_info["chosen_rewards"],
        #     loss_info["rejected_rewards"],
        # )
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            loss = loss * self.args.rpo_alpha - policy_chosen_logps_avg


        prefix = "eval_" if train_eval == "eval" else ""
        
        # how often the policy model is better at choosing the right response
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        # # how much the policy model is better
        # metrics[f"{prefix}rewards/margins"] = (
        #     (chosen_rewards - rejected_rewards).mean().cpu()
        # )

        # the log probability that the model would generate the tokens of the rejected string
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()


        for k in loss_info.keys():
            if '_' in k:
                a,b = k.split('_', 1)
                k2 = f"{b}/{a}"
            else:
                k2 = k
            v = loss_info[k]
            if isinstance(v, torch.Tensor):
                v = v.mean().detach().cpu().item()
            metrics[f"{prefix}{k2}"] = float(v)

        if self.state.global_step % self.args.print_every == 0:

            retain_cosine = F.cosine_similarity(
                policy_chosen_hs, reference_chosen_hs, dim=-1
            ).mean()
            rr_cosine = F.cosine_similarity(
                policy_rejected_hs, reference_chosen_hs, dim=-1
            ).mean()
            print(
                self.state.global_step,
                f"retain_cos_sim: {retain_cosine:.4f}. rr_cos_sim: {rr_cosine:.4f}",
            )
            metrics[f"{prefix}retain_cosine"] = retain_cosine
            metrics[f"{prefix}rr_cosine"] = rr_cosine

            print({k: f"{v:.2g}" for k, v in metrics.items()})
        
        return loss.mean(), metrics

    def reprpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_hs: torch.FloatTensor,
        policy_rejected_hs: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_chosen_hs: torch.FloatTensor,
        chosen_attn_mask: torch.BoolTensor,
        rejected_attn_mask: torch.BoolTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor(
                [0], dtype=pi_logratios.dtype, device=pi_logratios.device
            )
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        # log(prob_chosen/prob_rejected) the prob of the chosen strings over the rejected string. 0 is not difference. -ve means rejected is larger
        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # Can we weight by how much better the reference model was
        T = 30
        weight_correct = torch.softmax(-logits * T, 0).detach()



        # mean of bad repr should be more similar to the mean of good behavior
        loss_rr = sum_squared_error(policy_rejected_hs, reference_chosen_hs)
        loss_rr = mean_with_attention(loss_rr, rejected_attn_mask*chosen_attn_mask)
        loss_rr = wmean(loss_rr, 1 - weight_correct).mean()

        #  b l t h
        # a mean, norm, loss over the hidden dim of each layer
        # This loss says the good repr should be retained, weighted by how good this samples was
        loss_retain = sum_squared_error(
            policy_chosen_hs, reference_chosen_hs
        )
        loss_retain = mean_with_attention(loss_retain, chosen_attn_mask)
        loss_retain = wmean(loss_retain, weight_correct).mean()

        c_retain, c_rr = self.get_coeff()
        loss = (loss_rr * c_rr + loss_retain * c_retain * self.alpha).sum()

        # difference in logps for chosen responses, between policy and reference model
        # # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device)
                - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )

        # difference in logps for rejected responses, between policy and reference model
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        loss_dict = dict(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            loss_retain=loss_retain.detach(),
            loss_rr=loss_rr.detach(),
            pi_logratios=pi_logratios.detach(),
            ref_logratios=ref_logratios.detach(),
            weighting=weight_correct.mean(),
            logits=logits.mean().detach(),
            loss_component_rr = (loss_rr * c_rr).detach().mean(),
            loss_component_retain = loss_retain * c_retain * self.alpha,
            c_rr=c_rr,
            c_retain=c_retain,
        )

        # now mean any with ndim>0, and detach an cpu
        loss_dict = {k: v.mean().detach().cpu().item() for k, v in loss_dict.items()}

        return loss, loss_dict



def normalize_output(x):
    """
    if it's a tensor, detach and move to cpu, then to numpy
    if ndim>0, mean
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    if hasattr(x, 'ndim') and x.ndim > 0:
        x = x.mean()
    return x * 1.0 # to float
