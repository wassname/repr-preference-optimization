import torch
import torch.nn.functional as F
from reprpo.interventions.pl_base import PL_MODEL
from dataclasses import dataclass
from reprpo.interventions.config import ExperimentConfig
from .losses.helpers import cross_entropy_loss
from .helpers import compute_logprobs
from .dpo import compute_dpo_loss

# https://huggingface.co/docs/peft/developer_guides/custom_models
from torch import nn
from typing import Optional
from loguru import logger
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear

from torch.autograd import Function
from .dpo_projgrad import hs_norm, ProjGradHooks, compute_gradproj_loss_batch


class ProjBPHooks(ProjGradHooks):
    def register_hooks(self):
        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            module.register_full_backward_pre_hook(self.backward_prehook_projgrad)
            module.register_forward_hook(self.forward_hook_projgrad)
        logger.debug(f"ProjGrad Registered {len(modules)} hooks. {modules.keys()}")


    def backward_prehook_projgrad(self, module, grad_output):
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook

        ref_cho = module._cache['ref_cho']
        if isinstance(ref_cho, tuple):
            ref_cho = ref_cho[0]
        
        ref_rej = module._cache['ref_rej']
        if isinstance(ref_rej, tuple):
            ref_rej = ref_rej[0]
        
        # sum over tokens either using mask [b t h] -> [b h]
        mask = module._cache['ref_cho_mask'].unsqueeze(-1)
        ref_cho = (ref_cho * mask).sum(1) / mask.sum(1)
        mask = module._cache['ref_rej_mask'].unsqueeze(-1)
        ref_rej = (ref_rej * mask).sum(1) / mask.sum(1)

        # now take only the preferred direction
        eps = 1e-12
        preference_dir = ref_cho-ref_rej
        if self.reverse_pref:
            preference_dir = -preference_dir
        pref_dir_unit = preference_dir / preference_dir.norm(dim=-1, keepdim=True).clamp(eps)
        pref_dir_unit = pref_dir_unit.unsqueeze(1)
        
        # get projection of `grad` along ref_dir
        # grad is 'b t h'
        def proj_grad(grad):
            assert torch.isfinite(grad).all()

            # mag_clip, similar to PPO, we limit the update in hs-space. Simlar to PPO which does it per logit, we do it per hs
            if self.mag_clip is not None:
                # per sample
                # TODO check dims
                # preference_dir [b, h]
                # grad [b t h]
                ratios = grad.mean(1).norm(dim=-1)/((preference_dir).norm(dim=-1).clamp(eps)*self.mag_clip)
                ratios = ratios.clamp(1, None)[:, None, None]
                grad = grad / ratios
                assert torch.isfinite(grad).all()
            
            grad_proj_onto_pref = (pref_dir_unit * grad).sum(dim=-1, keepdim=True) * pref_dir_unit
            grad_orthogonal = grad - grad_proj_onto_pref

            # scale sideway movement so it's a proportion of prefered direction movement
            if self.scale_orth:
                scale = (grad_orthogonal).norm(dim=-1).clamp(eps) / (grad_proj_onto_pref).norm(dim=-1).clamp(eps) 
                scale = scale.unsqueeze(-1)
            else:
                scale = 1.0
            grad2 = F.leaky_relu(grad_proj_onto_pref, negative_slope=self.neg_slope) + self.β * grad_orthogonal * scale

            # keep the original magnitude
            assert torch.isfinite(grad2).all()
            # scale = torch.norm_except_dim(grad, 0).clamp(eps) / torch.norm_except_dim(grad2, 0).clamp(eps)
            scale = grad.norm().clamp(eps) / grad2.norm().clamp(eps)
            assert torch.isfinite(scale).all()
            # print(f"{grad.norm().item():.2f} -> {grad2.norm().item():.2f} -> {(grad2 * scale).norm().item():.2f}")
            grad2 = grad2 * scale
            assert torch.isfinite(grad2).all()

            return grad2
    
        res = tuple(proj_grad(g) for g in grad_output)
        # print(2,module)
        # print(3,[aa.shape for aa in grad_output], [aa.shape for aa in res])

        return res




class PL_GradBP_MODEL(PL_MODEL):
    def __init__(self, *args, β, reverse_pref, scale_orth, neg_slope, mag_clip, **kwargs):
        super().__init__(*args, **kwargs)
        self.projgrad = ProjBPHooks(self._model, β=β, reverse_pref=reverse_pref, scale_orth=scale_orth, neg_slope=neg_slope, mag_clip=mag_clip)

    def _loss_fn(self, batch, model):
        return compute_gradproj_loss_batch(batch, model, self.projgrad) 
    

@dataclass
class ProjBPConfig(ExperimentConfig):
    """
    Project gradient onto a preference direction during backprop. This means that later layer will change the gradint for earleir layers.
    """
    lr: float = 1e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    β: float = 0.2

    reverse_pref: bool = False
    """
    reverse the preference direction
    """

    scale_orth: bool = False
    """
    scale the orthogonal movement to be β proportion of the prefered direction movement
    """

    neg_slope: float = 0.8
    """
    When clipping the gradient in the negative preference direction, we can use leakly relu with this slope. 0 is relu. 0.01 is leaky relu.
    """

    mag_clip: Optional[float] = 0.6
    """
    Clip the magnitude of the gradient in the hs space, to ensure a proximal policy in hs space. Value is a fraction of the distance of the preference vector
    """

    _cls = PL_GradBP_MODEL

    _model_keys = ["lr", "β", "reverse_pref", "scale_orth", "neg_slope", "mag_clip",]

    @property
    def _name(self):
        return "projbp"
