import torch
from torch import nn
import torch.nn.functional as F

# https://huggingface.co/docs/peft/developer_guides/custom_models
from loguru import logger
from peft.tuners.tuners_utils import BaseTunerLayer

from dataclasses import dataclass
from contextlib import contextmanager

from typing import Optional
from einops import reduce, rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from reprpo.interventions.pl_base import PL_MODEL
from reprpo.interventions.config import ExperimentConfig
from .dpo_helpers import cross_entropy_loss
from .dpo_helpers import compute_logprobs
from .dpo import compute_dpo_loss


def _norm(x, dims):
    return (x ** 2).sum(dims).sqrt()

def hs_norm(t: Float[Tensor, 'b t h']) -> Float[Tensor, 'b t h']:
    # we mean over tokens, as the dimensions are not comparable
    # and norm over hs
    t = reduce(t, 'b t h -> b h', 'mean')
    t = reduce(t, 'b h -> b', _norm)
    return repeat(t, 'b -> b t h', t=1, h=1)

class ProjGradHooks:
    def __init__(self, model, β=0.1, reverse_pref=False, scale_orth=False, neg_slope=0.0, mag_clip=None, weight_dim=2, use_pref_ref=True):
        self.model = model
        self.β = β
        self.reverse_pref = reverse_pref
        self.scale_orth = scale_orth
        self.neg_slope = neg_slope
        self.mag_clip = mag_clip
        self.weight_dim = weight_dim
        self.use_pref_ref = use_pref_ref
        self.register_hooks()

    def enabled_modules(self):
        """Yield peft modules that have requires_grad=True"""
        modules = {}
        for module_name, module in self.model.named_modules():

            # only peft layers
            if not isinstance(module, (BaseTunerLayer,)):
                continue

            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    modules[module_name] = module
                    break

        return modules      

    def register_hooks(self):
        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            # module.register_full_backward_pre_hook(self.backward_prehook_projgrad)
            module.register_forward_hook(self.forward_hook_projgrad)
        logger.debug(f"ProjGrad Registered {len(modules)} hooks. {modules.keys()}")


    def forward_hook_projgrad(self, m, inputs, output):
        if not hasattr(m, "_cache"):
            m._cache = {}
        
        if m._mode in ["ref_cho", "ref_rej"]:
            output2 = tuple(o.detach() for o in output)
            m._cache[m._mode+'_mask'] = m._mask.detach()
            m._cache[m._mode] = output2 # (o.detach() for o in output)
        elif m._mode in ["cho", "rej"]:
            output2 = tuple(o.detach() for o in output)
            assert m._mode in ["cho", "rej"]
            m._cache[m._mode+'_mask'] = m._mask.detach()
            m._cache[m._mode] = output2 # (o.detach() for o in output)
        elif m._mode == "normal":
            pass
        elif m._mode == "clear":
            m._cache = {}
        else:
            raise ValueError(f"Invalid mode {m._mode}")
        return output


    @contextmanager
    def set_projgrad_mode(self, mode: str, mask=None):

        # TODO consider temp hooks https://github.com/huggingface/peft/blob/ccc350151f95a9ff95da046bae5671da75eab52f/src/peft/tuners/lora/model.py#L438 https://github.com/davidbau/baukit/blob/main/baukit/nethook.py
        # but it wont work with ba

        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            module._old_mode = getattr(module, '_mode', "normal") + ""
            module._mode = mode
            module._mask = mask
        yield self.model
        for k, module in modules.items():
            module._mode = module._old_mode
            module._mask = None

def compute_gradproj_loss_batch(batch, model, projgrad, β=0.1, use_policy_weights=False):
    """Compute the DPO loss on an input batch"""

    model_kwargs = dict(
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    with projgrad.set_projgrad_mode("clear"):
        pass

    model.eval()
    with projgrad.set_projgrad_mode("ref_cho", batch["chosen_mask"]):
        with model.disable_adapter():
            with torch.no_grad():
                ref_cho = model(
                    batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs
                )
                ref_cho_logp = compute_logprobs(
                    logits=ref_cho.logits,
                    input_ids=batch["chosen"],
                    selection_mask=batch["chosen_mask"] * (1-batch['prompt_mask']),
                )
    with projgrad.set_projgrad_mode("ref_rej", batch["rejected_mask"]):
        with model.disable_adapter():
            with torch.no_grad():
                ref_rej = model(
                    batch["rejected"], attention_mask=batch["rejected_mask"], **model_kwargs
                )
                ref_rej_logp = compute_logprobs(
                    logits=ref_rej.logits,
                    input_ids=batch["rejected"],
                    selection_mask=batch["rejected_mask"] * (1-batch['prompt_mask']),
                )

    model.train()
    with projgrad.set_projgrad_mode("cho", batch["chosen_mask"]):
        pi_cho = model(batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs)
    pi_cho_logp = compute_logprobs(
        logits=pi_cho.logits,
        input_ids=batch["chosen"],
        selection_mask=batch["chosen_mask"] * (1-batch['prompt_mask']),
    )
    with projgrad.set_projgrad_mode("rej", batch["rejected_mask"]):
        pi_rej = model(
            batch["rejected"], attention_mask=batch["rejected_mask"], **model_kwargs
        )
    pi_rej_logp = compute_logprobs(
        logits=pi_rej.logits,
        input_ids=batch["rejected"],
        selection_mask=batch["rejected_mask"] * (1-batch['prompt_mask']),
    )

    # loss, info = compute_dpo_loss(
    #     model_chosen_logprobs=pi_cho_logp['label_logp'],
    #     model_rejected_logprobs=pi_rej_logp['label_logp'],
    #     reference_chosen_logprobs=ref_cho_logp['label_logp'],
    #     reference_rejected_logprobs=ref_rej_logp['label_logp'],
    #     β=β,
    # )

    loss, info = compute_dpo_loss_batch_plus(
        batch,
        pi_cho,
        pi_rej,
        ref_cho,
        ref_rej,
        β=β,
        use_policy_weights=use_policy_weights,
    )

    if use_policy_weights:
        policy_weights = torch.clamp(
            torch.exp(pi_cho_logp['log_policy_weights'] + pi_rej_logp['log_policy_weights']),
            max=1
        )

        loss = loss * policy_weights

    with torch.no_grad():

        nll_loss = info["nll_loss"] = cross_entropy_loss(
            pi_cho.logits, batch["chosen"], batch["chosen_mask"]
        ).mean()
        ref_nll_loss = info["ref_nll_loss"] = cross_entropy_loss(
            ref_cho.logits, batch["chosen"], batch["chosen_mask"]
        ).mean()
        info["nll_loss_ratio"] = (nll_loss / ref_nll_loss).mean()

    return loss, info


class PL_ProjGrad_MODEL(PL_MODEL):
    def __init__(self, *args, β, reverse_pref, scale_orth, neg_slope, mag_clip, weight_dim, use_pref_ref, use_policy_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.projgrad = ProjGradHooks(self._model, β=β, reverse_pref=reverse_pref, scale_orth=scale_orth, neg_slope=neg_slope, mag_clip=mag_clip, weight_dim=weight_dim, use_pref_ref=use_pref_ref)
        self.use_policy_weights = use_policy_weights

    def _loss_fn(self, batch, model):
        return compute_gradproj_loss_batch(batch, model, self.projgrad, use_policy_weights=self.use_policy_weights)
    
    def on_after_backward(self):
        """
        Here we clip the gradient on each layer to only move in the preference direction in hidden-state space

        on_after_backward: Called in the training loop after loss.backward() and before optimizers do anything. 
        """
        proj_frac = []
        nproj_frac = []
        h = self.projgrad

        for k, module in self.projgrad.enabled_modules().items():

            if self.projgrad.use_pref_ref:
                ref_cho = module._cache['ref_cho']                
                ref_rej = module._cache['ref_rej']                
                # sum over tokens either using mask
                # [b t h] -> [b h]
                mask_cho = module._cache['ref_cho_mask'].unsqueeze(-1)
                mask_rej = module._cache['ref_rej_mask'].unsqueeze(-1)
            else:
                ref_cho = module._cache['cho']                
                ref_rej = module._cache['rej']      
                mask_cho = module._cache['cho_mask'].unsqueeze(-1)
                mask_rej = module._cache['rej_mask'].unsqueeze(-1)

            if isinstance(ref_cho, tuple):
                ref_cho = ref_cho[0]
            
            if isinstance(ref_rej, tuple):
                ref_rej = ref_rej[0]
            
            ref_cho = (ref_cho * mask_cho).sum(1) / mask_cho.sum(1)
            ref_rej = (ref_rej * mask_rej).sum(1) / mask_rej.sum(1)
            
            eps = 1e-12
            # note these have shape [batch, hs_dim] at this point
            preference_dir = (ref_cho - ref_rej).mean(0) # FIXME ideally we do it per sample, but that would require modifying param.grad when it's applied after backprop
            if h.reverse_pref:
                preference_dir = -preference_dir
            pref_dir_unit = preference_dir / preference_dir.norm(dim=0).clamp(eps)
            pref_dir_unit = pref_dir_unit.unsqueeze(0) # for the other dim of the weights

            for param_name, param in module.named_parameters():
                if param.grad is None:
                    continue

                hs_dim = preference_dir.shape[0]
                if (param.grad.shape[0] == hs_dim) and (h.weight_dim in [0, 2]):
                    pref_dir_unit = pref_dir_unit.T
                    dim = 0
                    # print('flip')
                elif (param.grad.shape[-1] == hs_dim)  and (h.weight_dim in [1, 2]):
                    dim=-1
                else:
                    # print(f"Skipping {k} {param_name} shape={param.grad.shape} pref_dir_unit.shape={pref_dir_unit.shape}")
                    continue
                
                grad = param.grad.clone()

                # mag_clip, similar to PPO, we limit the update in hs-space. Simlar to PPO which does it per logit, we do it per hs
                if h.mag_clip is not None:
                    # per sampler
                    # grad shape [in_dim, hs] or [hs, out_dim]
                    # pref shapee [hs]
                    other_dim = int(not bool(dim))
                    ratios = grad.mean(other_dim)/(preference_dir.clamp(eps)*h.mag_clip)
                    ratios = ratios.clamp(1, None).unsqueeze(other_dim)
                    grad = grad / ratios
                
                grad_proj_onto_pref = (pref_dir_unit * grad).sum(dim=dim, keepdim=True) * pref_dir_unit
                grad_orthogonal = grad - grad_proj_onto_pref

                # scale sideway movement so it's a proportion of prefered direction movement
                if h.scale_orth:
                    # these are the same shape
                    scale = (grad_orthogonal).norm(dim=dim).clamp(eps) / (grad_proj_onto_pref).norm(dim=dim).clamp(eps) 
                    scale = scale.unsqueeze(dim)
                else:
                    scale = 1.0

                param.grad = F.leaky_relu(grad_proj_onto_pref, h.neg_slope) + h.β * grad_orthogonal / scale

                proj_frac.append(F.relu(grad_proj_onto_pref).norm()/grad.norm())
                nproj_frac.append(F.relu(-grad_proj_onto_pref).norm()/grad.norm())
        if len(proj_frac) > 0:
            self.log_dict({
                "train/proj_frac": torch.stack(proj_frac).mean(),
                "train/nproj_frac": torch.stack(nproj_frac).mean(),
                }, on_step=True)


@dataclass
class ProjGradConfig(ExperimentConfig):
    """
    This is DPO, with a hard constrain on the gradient, so only move in the preference direction in hidden-state space

    It takes all the Lora layers, and on the backward pass it clip the gradient to only move in the cho_hs-ref_hs vector
    """

    lr: float = 6e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    β: float = 0.8
    """multiplier on the orthogonal movement"""

    reverse_pref: bool = True
    """
    reverse the preference direction
    """

    weight_dim: int = 1
    """
    1 for forward, 0 for back, 2 for both
    """

    scale_orth: bool = False
    """
    scale the orthogonal movement to be β proportion of the prefered direction movement
    """

    neg_slope: float = 0.
    """
    When clipping the gradient in the negative preference direction, we can use leakly relu with this slope. 0 is relu. 0.01 is leaky relu.
    """

    mag_clip: Optional[float] = None
    """
    Clip the magnitude of the gradient in the hs space, to ensure a proximal policy in hs space. Value is a fraction of the distance of the preference vector
    """

    use_pref_ref: bool = True
    """use the reference model to get the preference vector. Is false we use the moving policy and it could lead to a better outcome, or instability (TODO investigate)"""

    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""

    _cls = PL_ProjGrad_MODEL

    _model_keys = ["lr", "β", "reverse_pref", "scale_orth", "neg_slope", "mag_clip", "weight_dim", "use_pref_ref"]

    @property
    def _name(self):
        return "projgrad"
