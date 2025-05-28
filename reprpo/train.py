#!/usr/bin/env python3
"""
Simplified InnerPO Training Script

This script implements both DPO and InnerPO (Inner Preference Optimization) in a clean, minimal way.
No complex frameworks - just PyTorch, transformers, and simple command-line arguments.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, 
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm
import wandb
import tyro

# Local imports  
from .data_loader import load_preference_dataset

# Import real transforms - simplified imports to avoid dependency issues
try:
    from .interventions.transforms.supressed import SupressedHSTransform
    from .interventions.transforms.ether import EtherTransforms  
    from .interventions.transforms.none import NoneTransforms
    HAS_REAL_TRANSFORMS = True
except ImportError as e:
    print(f"Warning: Could not import real transforms ({e}), using simplified versions")
    HAS_REAL_TRANSFORMS = False


class Method(Enum):
    """Training methods available"""
    DPO = "dpo"
    PROJGRAD = "projgrad"
    INNERPO = "innerpo"


class Transform(Enum):
    """Hidden state transforms for InnerPO"""
    NONE = "none"          # No transform
    SUPPRESSED = "supr"    # Remove mean (suppressed hidden states)
    ETHER = "ether"        # ETHER orthogonal transform  
    ORTHO = "ortho"        # Simple orthogonal projection


class Dataset(Enum):
    """Available datasets"""
    MATH = "math"
    CODE = "code"
    ALPACA_MMLU = "alpaca_mmlu"
    COOKING = "cooking"
    ALPACA_LOW_QUALITY = "alpaca_low_quality"
    MATHS_EASY = "maths_easy"


@dataclass
class Config:
    """Modular configuration - mix and match components as needed"""
    
    # === Core Method ===
    method: Method = Method.DPO
    transform: Transform = Transform.NONE  # Only used if method=INNERPO
    
    # === Model & Data ===
    base_model: str = "Qwen/Qwen3-0.6B"
    dataset: Dataset = Dataset.MATH
    
    # === Training ===
    seed: int = 1
    lr: float = 1e-5
    batch_size: int = 4
    max_steps: int = 500
    max_length: int = 512
    
    # === LoRA ===
    lora_r: int = 16
    lora_alpha: int = 32
    
    # === Loss Parameters ===
    beta: float = 0.1           # DPO beta parameter
    innerpo_alpha: float = 0.1  # InnerPO loss weight (only for INNERPO method)
    
    # === InnerPO Layers ===
    layers: Optional[List[int]] = None  # Layers to apply InnerPO to
    
    # === System ===
    device: str = "auto"
    output_dir: str = "./outputs"
    verbose: int = 1
    
    def __post_init__(self):
        # Convert enum values to strings for compatibility
        if isinstance(self.method, Method):
            self.method = self.method.value
        if isinstance(self.transform, Transform):
            self.transform = self.transform.value  
        if isinstance(self.dataset, Dataset):
            self.dataset = self.dataset.value
            
        # Set default layers if not specified
        if self.layers is None:
            self.layers = [8, 12, 16]  # Default middle layers
            
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def method_name(self) -> str:
        """Get a descriptive name for the current method configuration"""
        if self.method == "dpo":
            return "dpo"
        elif self.method == "projgrad":
            return "projgrad"
        elif self.method == "innerpo":
            return f"innerpo-{self.transform}"
        else:
            return f"{self.method}-{self.transform}"


class SimpleDataset(TorchDataset):
    """Minimal preference dataset"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: {"prompt": str, "chosen": str, "rejected": str}
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)["input_ids"]
        
        # Combine prompt + response
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens
        
        # Truncate if needed
        chosen_full = chosen_full[:self.max_length]
        rejected_full = rejected_full[:self.max_length]
        
        return {
            "chosen_ids": torch.tensor(chosen_full, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_full, dtype=torch.long),
            "prompt_len": len(prompt_tokens)
        }


def collate_fn(batch):
    """Simple collate function with padding"""
    chosen_ids = [item["chosen_ids"] for item in batch]
    rejected_ids = [item["rejected_ids"] for item in batch]
    prompt_lens = [item["prompt_len"] for item in batch]
    
    # Pad sequences
    max_len = max(max(len(x) for x in chosen_ids), max(len(x) for x in rejected_ids))
    
    chosen_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    rejected_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (chosen, rejected) in enumerate(zip(chosen_ids, rejected_ids)):
        chosen_padded[i, :len(chosen)] = chosen
        rejected_padded[i, :len(rejected)] = rejected
    
    return {
        "chosen_ids": chosen_padded,
        "rejected_ids": rejected_padded,
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long)
    }


def create_transform(transform_name: str, model, dim_sizes: Dict[str, int]):
    """Create transform instance from name"""
    if not HAS_REAL_TRANSFORMS:
        print(f"Using simplified {transform_name} transform")
        return None  # Will use apply_transform fallback
        
    if transform_name == "none":
        return NoneTransforms(dim_sizes, model)
    elif transform_name == "supr":
        return SupressedHSTransform(dim_sizes, model)
    elif transform_name == "ether":
        return EtherTransforms(dim_sizes, model)
    else:
        # Fallback to none transform
        return NoneTransforms(dim_sizes, model)


def apply_transform(hidden_states: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the specified transform to hidden states (simplified fallback)"""
    if transform == "none":
        return hidden_states
    elif transform == "supr":
        # Simplified suppressed: remove mean component
        return hidden_states - hidden_states.mean(dim=-1, keepdim=True)
    elif transform == "ether":
        # Simplified ETHER: normalize
        return F.normalize(hidden_states, dim=-1)
    elif transform == "ortho":
        # Simplified orthogonal: normalize
        return F.normalize(hidden_states, dim=-1)
    else:
        return hidden_states


def get_hidden_states(model, input_ids, layers: List[int]) -> Dict[int, torch.Tensor]:
    """Extract hidden states from specified layers"""
    hidden_states = {}
    
    def hook_fn(layer_idx):
        def hook(module, input_tensor, output):
            # output[0] is the hidden state (input_tensor unused but required for hook signature)
            hidden_states[layer_idx] = output[0].detach()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in layers:
        if layer_idx < len(model.model.layers):  # Assuming transformer structure
            hook = model.model.layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return hidden_states


def dpo_loss(model, ref_model, chosen_ids, rejected_ids, prompt_lens, beta: float = 0.1):
    """Standard DPO loss"""
    # Get logprobs for chosen and rejected
    chosen_logits = model(chosen_ids).logits
    rejected_logits = model(rejected_ids).logits
    
    ref_chosen_logits = ref_model(chosen_ids).logits
    ref_rejected_logits = ref_model(rejected_ids).logits
    
    # Calculate log probabilities for response tokens only
    chosen_logprobs = []
    rejected_logprobs = []
    ref_chosen_logprobs = []
    ref_rejected_logprobs = []
    
    for i, prompt_len in enumerate(prompt_lens):
        # Response tokens start after prompt
        response_start = prompt_len
        
        # Chosen
        chosen_response_logits = chosen_logits[i, response_start-1:-1]  # Shift by 1
        chosen_response_ids = chosen_ids[i, response_start:]
        chosen_logprob = F.log_softmax(chosen_response_logits, dim=-1)
        chosen_logprob = torch.gather(chosen_logprob, -1, chosen_response_ids.unsqueeze(-1))
        chosen_logprobs.append(chosen_logprob.squeeze(-1).sum())
        
        # Reference chosen
        ref_chosen_response_logits = ref_chosen_logits[i, response_start-1:-1]
        ref_chosen_logprob = F.log_softmax(ref_chosen_response_logits, dim=-1)
        ref_chosen_logprob = torch.gather(ref_chosen_logprob, -1, chosen_response_ids.unsqueeze(-1))
        ref_chosen_logprobs.append(ref_chosen_logprob.squeeze(-1).sum())
        
        # Rejected
        rejected_response_logits = rejected_logits[i, response_start-1:-1]
        rejected_response_ids = rejected_ids[i, response_start:]
        rejected_logprob = F.log_softmax(rejected_response_logits, dim=-1)
        rejected_logprob = torch.gather(rejected_logprob, -1, rejected_response_ids.unsqueeze(-1))
        rejected_logprobs.append(rejected_logprob.squeeze(-1).sum())
        
        # Reference rejected
        ref_rejected_response_logits = ref_rejected_logits[i, response_start-1:-1]
        ref_rejected_logprob = F.log_softmax(ref_rejected_response_logits, dim=-1)
        ref_rejected_logprob = torch.gather(ref_rejected_logprob, -1, rejected_response_ids.unsqueeze(-1))
        ref_rejected_logprobs.append(ref_rejected_logprob.squeeze(-1).sum())
    
    chosen_logprobs = torch.stack(chosen_logprobs)
    rejected_logprobs = torch.stack(rejected_logprobs)
    ref_chosen_logprobs = torch.stack(ref_chosen_logprobs)
    ref_rejected_logprobs = torch.stack(ref_rejected_logprobs)
    
    # DPO loss
    logits = (chosen_logprobs - rejected_logprobs) - (ref_chosen_logprobs - ref_rejected_logprobs)
    loss = -F.logsigmoid(beta * logits).mean()
    
    return loss


def innerpo_loss(model, ref_model, chosen_ids, rejected_ids, prompt_lens, 
                 layers: List[int], transform: str, beta: float = 0.1, alpha: float = 0.1):
    """InnerPO loss - DPO + hidden state alignment"""
    # Get DPO loss first
    dpo_loss_val = dpo_loss(model, ref_model, chosen_ids, rejected_ids, prompt_lens, beta)
    
    # Get hidden states from current model
    chosen_hidden = get_hidden_states(model, chosen_ids, layers)
    rejected_hidden = get_hidden_states(model, rejected_ids, layers)
    
    # Get reference hidden states
    ref_chosen_hidden = get_hidden_states(ref_model, chosen_ids, layers)
    ref_rejected_hidden = get_hidden_states(ref_model, rejected_ids, layers)
    
    # InnerPO loss: align hidden states in preference direction
    innerpo_loss_val = 0.0
    for layer_idx in layers:
        if layer_idx in chosen_hidden and layer_idx in rejected_hidden:
            # Preference direction from reference model
            pref_dir = ref_chosen_hidden[layer_idx] - ref_rejected_hidden[layer_idx]
            
            # Current model's hidden state difference
            current_diff = chosen_hidden[layer_idx] - rejected_hidden[layer_idx]
            
            # Apply transform
            pref_dir = apply_transform(pref_dir, transform)
            current_diff = apply_transform(current_diff, transform)
            
            # Cosine similarity loss (want them to align)
            cos_sim = F.cosine_similarity(
                current_diff.view(-1, current_diff.size(-1)),
                pref_dir.view(-1, pref_dir.size(-1)),
                dim=-1
            )
            innerpo_loss_val += (1 - cos_sim).mean()
    
    total_loss = dpo_loss_val + alpha * innerpo_loss_val
    return total_loss


def projgrad_loss(model, ref_model, chosen_ids, rejected_ids, prompt_lens, 
                  layers: List[int], beta: float = 0.1):
    """Projection gradient loss - simplified version"""
    # For now, just use DPO loss - can be extended with gradient projection
    return dpo_loss(model, ref_model, chosen_ids, rejected_ids, prompt_lens, beta)


def main():
    # Use tyro for clean CLI with dataclass support
    config = tyro.cli(Config)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed) 
    torch.manual_seed(config.seed)
    
    print("This is the standalone train.py - use the CLI or Lightning version for training")


if __name__ == "__main__":
    main()