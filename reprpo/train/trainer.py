from dataclasses import dataclass
from einops import rearrange, repeat
from trl import DPOConfig, DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import gc
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    PreTrainedModel,
)
from trl.trainer.utils import (
    pad_to_length,
)
from reprpo.helpers.svd_decomposer import SVDDecomposer, DualSVDDecomposer, SoftSVDDecomposer
from reprpo.gen import generation_test, get_model_generations


# def norm(a: Float[Tensor, 'b l t h']) ->  Float[Tensor, 'b l t h']:
#     """normalise the hidden states."""
#     im = rearrange(a, 'b l t h -> b t (l h)')
#     # print(torch.norm(im, dim=-1, keepdim=True, p=1)+1)
#     im = im / (torch.norm(np.abs(im), dim=-1, keepdim=True, p=2)+1)
#     im = rearrange(im, 'b t (l h) -> b l t h', l=a.shape[1], h=a.shape[-1])
#     return im

def normalize_per(a, norm_dims=(1, 2, 3), eps=1e-12):
    """normalize per norm_dims

    this means dividing by the norm of the other dims
    """
    # use F.normalize?
    return a / (torch.norm_except_dim(a, dim=norm_dims, p=2, keepdim=True) + eps)

def mult_with_attention(x: Float[Tensor, 'b l t h'], attn_mask: Float[Tensor, 'b t']) -> Float[Tensor, 'b l t h']:
    layer_attn_mask = repeat(attn_mask, 'b t -> b l t h', l=x.shape[1], h=1)
    return (x * layer_attn_mask)

def mean_with_attention(x: Float[Tensor, 'b l t h'], attn_mask: Float[Tensor, 'b t'], dim: int = 2) -> Float[Tensor, 'b l h']:
    """mean of x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, 'b t -> b l t h', l=x.shape[1], h=1).detach()
    return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)

def symlog(x, eps=1e-6):
    return torch.sign(x) * torch.log(torch.abs(x)+eps)


def symlog_loss(x, y, eps=1e-6):
    # maybe a simple norm or x and y the same here? just to scale the outputs
    e = symlog(x)-symlog(y)
    # e = norm_h(e)
    return e**2

def norm_error(input: Float[Tensor, 'b l t h'], target: Float[Tensor, 'b l t h']) -> Float[Tensor, 'b l t h']:
    # from https://github.com/GraySwanAI/circuit-breakers/blob/main/src/lorra_circuit_breaker.py
    return torch.norm(input - target, dim=-1, p=2, 
                      dtype=torch.float, 
                      keepdim=True)#.nanmean()

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
    quantile: float = 0.10
    dual_svd: bool = False

    # NOTE to self, do not pass both peft_config and model_adapter_name. peft_config creates a new adapter


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

        # convert
        if args.dual_svd:
            self.decomposer = DualSVDDecomposer(self.model.get_input_embeddings().weight.clone().float(), self.model.lm_head.weight.clone(), quantile=args.quantile)
        else:
            self.decomposer = SoftSVDDecomposer(self.model.lm_head.weight.clone().float(), quantile=args.quantile)


    def get_training_progress(self):
        # in the paper they claim it's schedule but they end up making it loop every 300 steps, but then use 1500 steps for the loss
        return self.state.global_step / self.num_training_steps
    
    def get_coeff(self):
        # training schedule... I would prefer to make it stepped so it was more obvious loss
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
        log_softmax=True
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

        if log_softmax:
            logits = logits.log_softmax(-1)

        per_token_logps = torch.gather(
            logits, dim=2, index=labels.unsqueeze(2)
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
            max_length=self.max_length
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
        # this includes prompt and padding
        hs = collect_hs(outs.hidden_states)[:, self.collection_layers]
        # del outs
        # gc.collect()

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
            # all_logps = torch.log(torch.exp(all_logps) / size_completion + 1e-12)
            # NOTE for some reason the model is still biased toward longer answers, even though this should neutralise it

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
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                    _,
                    ref_chosen_hs,
                    ref_rejected_hs,
                    _,
                    _
                ) = self.concatenated_forward(self.model, batch)
        ref_chosen_hs = ref_chosen_hs.detach()
        ref_chosen_logps = ref_chosen_logps.detach()
        ref_rejected_logps = ref_rejected_logps.detach()
        ref_rejected_hs = ref_rejected_hs.detach()

        model.train()
        (
            pi_chosen_logps,
            pi_rejected_logps,
            _,
            _,
            pi_chosen_logps_avg,
            pi_chosen_hs,
            pi_rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask
        ) = self.concatenated_forward(model, batch)

        loss, loss_info = self.reprpo_loss(
            pi_chosen_logps,
            pi_rejected_logps,
            pi_chosen_hs,
            pi_rejected_hs,
            ref_chosen_logps,
            ref_rejected_logps,
            ref_chosen_hs,
            ref_rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask
        )
        # losses, chosen_rewards, rejected_rewards, loss_retain, loss_rr = loss_info
        chosen_rewards, rejected_rewards = (
            loss_info["chosen_rewards"],
            loss_info["rejected_rewards"],
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            loss = loss * self.args.rpo_alpha - pi_chosen_logps_avg


        prefix = "eval_" if train_eval == "eval" else ""
        
        # how often the policy model is better at choosing the right response
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        # how much the policy model is better
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().cpu()
        )

        # the log probability that the model would generate the tokens of the rejected string
        metrics[f"{prefix}logps/rejected"] = pi_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = pi_chosen_logps.detach().mean().cpu()


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
                pi_chosen_hs, ref_chosen_hs, dim=-1
            ).mean()
            rr_cosine = F.cosine_similarity(
                pi_rejected_hs, ref_chosen_hs, dim=-1
            ).mean()
            print(
                self.state.global_step,
                f"retain_cos_sim: {retain_cosine:.4f}. rr_cos_sim: {rr_cosine:.4f}",
            )
            metrics[f"{prefix}retain_cosine"] = retain_cosine
            metrics[f"{prefix}rr_cosine"] = rr_cosine

            print({k: f"{v:.4g}" for k, v in metrics.items()})
        
        return loss.mean(), metrics
    
    def res_det(self, hs):
        """use SVG to decompose hs into the output and residual components"""
        return hs - self.decomposer(hs)#.detach() # FIXME, should I not detatch this?



    def reprpo_loss(
        self,
        pi_chosen_logps: torch.FloatTensor,
        pi_rejected_logps: torch.FloatTensor,
        pi_cho_hs: torch.FloatTensor,
        pi_rej_hs: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        ref_cho_hs: torch.FloatTensor,
        ref_rej_hs: torch.FloatTensor,
        cho_attn_mask: torch.BoolTensor,
        rej_attn_mask: torch.BoolTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            pi_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            pi_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            ref_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        pi_logratios = pi_chosen_logps - pi_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor(
                [0], dtype=pi_logratios.dtype, device=pi_logratios.device
            )
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        # log(prob_chosen/prob_rejected) the prob of the chosen strings over the rejected string. 0 is not difference. -ve means rejected is larger
        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios # was pi more likely to chose the correct response or the reference model

        # Can we weight by how much better the reference model was
        # in dpo we minimise it, so lower is better, here we are weighting it, so take the -ve to higher is more correct
        # NOTE: -logits is if pi is more correct than ref, and focuses on what model gets wrong, unstable, moving target
        # -ref_logratios is is the reference model lean toward correct, and is stable
        T = 2
        weight_correct = torch.softmax(-ref_logratios * T, 0).detach()

        def dist_w_attn_mask( chosen_hs, rejected_hs, attn):
            dist = rejected_hs - chosen_hs
            dist = mean_with_attention(dist, attn.detach())
            assert torch.isfinite(dist).all() # FIXME nans
            # loss_rr = symlog(loss_rr)
            # loss_rr = wmean(loss_rr, 1 - weight_correct)
            return (dist**2).nanmean()        

        comb_attn_mask = cho_attn_mask * rej_attn_mask

        # decompose the hidden state differences into the part that is input or output by the embedding or unembedding wieghts, then only concern outselves with the reaminder (which may contain internal only information such as style or concepts)
        svd_dist_cho2rej_pi2ref = dist_w_attn_mask(
            self.res_det(ref_cho_hs).detach(), 
            self.res_det(pi_rej_hs),
            comb_attn_mask
        )

        # the loss is small, express it as a fraction of the reference values
        svd_dist_cho2rej_ref2ref = dist_w_attn_mask(
            self.res_det(ref_cho_hs), 
            self.res_det(ref_rej_hs), 
            comb_attn_mask
        )

        # how much we've reduced the distance between the chosen and rejected responses, compared to reference model
        loss_reroute = svd_dist_cho2rej_pi2ref / svd_dist_cho2rej_ref2ref.nanmean().detach()

        # this loss measures how much the policy model has retained the information in the chosen responses, compared to the reference model
        hs_dist_cho2cho_pi2ref = dist_w_attn_mask(
            pi_cho_hs, 
            ref_cho_hs.detach(), 
            cho_attn_mask
        )

        # scale it, so that it's expressed as a fraction of the dist between rej2cho on the ref model
        hs_dist_cho2rej_ref2ref = dist_w_attn_mask(
            ref_cho_hs, 
            ref_rej_hs, 
            comb_attn_mask
        )
        # +1 so it start on par with reroute loss, and we can see it diverge?? TODO revisit
        loss_retain = hs_dist_cho2cho_pi2ref / hs_dist_cho2rej_ref2ref.nanmean().detach() + 1

        # Weightings
        c_retain, c_reroute = self.get_coeff()
        c_reroute = c_retain = 1
        loss = (loss_reroute.nanmean() * c_reroute + loss_retain.nanmean() * c_retain * self.alpha)

        # difference in logps for chosen responses, between policy and reference model
        # # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        chosen_rewards = (
            self.beta
            * (
                pi_chosen_logps.to(self.accelerator.device)
                - ref_chosen_logps.to(self.accelerator.device)
            ).detach()
        )

        # difference in logps for rejected responses, between policy and reference model
        rejected_rewards = (
            self.beta
            * (
                pi_rejected_logps.to(self.accelerator.device)
                - ref_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        loss_dict = dict(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            loss_retain=loss_retain.detach(),
            loss_reroute=loss_reroute.detach(),
            pi_logratios=pi_logratios.detach(),
            ref_logratios=ref_logratios.detach(),
            weighting=weight_correct.mean(),
            logits=logits.mean().detach(),
            loss_component_rr = (loss_reroute * c_reroute).detach().mean(),
            loss_component_retain = (loss_retain * c_retain * self.alpha).detach().mean(),
            c_rr=c_reroute,
            c_retain=c_retain,
        )

        loss_dict = {k: normalize_output(v) for k, v in loss_dict.items()}

        return loss, loss_dict
    

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        # note subclassed to ensure max length each time
        concatenated_batch = {}

        if is_encoder_decoder:
            raise NotImplementedError("Encoder-decoder models are not supported yet.")
        else:
            pass
            # max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_vision_model:
            raise NotImplementedError("Vision models are not supported yet.")
        return concatenated_batch
    

    def evaluate(self, eval_dataset=None, **kwargs):
        self.model.eval()
        get_model_generations(self.model, self.tokenizer, N=1, max_new_tokens=64)
        # TODO also consider a propr DPO accurac on OOS and IS ds
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)




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
