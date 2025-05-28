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
from data_loader import load_preference_dataset


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


def apply_suppressed_transform(hidden_states: torch.Tensor) -> torch.Tensor:
    """Suppressed Hidden States - remove the mean component"""
    # This removes the "bias" or common component across all positions
    return hidden_states - hidden_states.mean(dim=-1, keepdim=True)


def apply_ether_transform(hidden_states: torch.Tensor) -> torch.Tensor:
    """ETHER transform - orthogonal projection that preserves task-relevant information"""
    # ETHER (from the paper) projects onto a subspace that removes irrelevant directions
    # Simplified implementation: SVD-based projection
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Flatten to (batch * seq, hidden_dim) for SVD
    flattened = hidden_states.view(-1, hidden_dim)
    
    # Mean center
    mean_centered = flattened - flattened.mean(dim=0, keepdim=True)
    
    # SVD to find principal components
    try:
        U, S, V = torch.svd(mean_centered)
        # Keep top 80% of components (heuristic)
        k = int(0.8 * hidden_dim)
        # Project onto top-k subspace
        projected = torch.mm(torch.mm(mean_centered, V[:, :k]), V[:, :k].t())
        return projected.view(batch_size, seq_len, hidden_dim)
    except (RuntimeError, torch.linalg.LinAlgError):
        # Fallback to normalization if SVD fails
        return F.normalize(hidden_states, dim=-1)


def apply_ortho_transform(hidden_states: torch.Tensor) -> torch.Tensor:
    """Simple orthogonal projection - projects onto orthogonal subspace"""
    # QR decomposition for orthogonalization
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Reshape for QR
    flattened = hidden_states.view(-1, hidden_dim)
    
    try:
        # QR decomposition to get orthogonal basis
        Q, R = torch.qr(flattened.t())
        # Project onto orthogonal subspace
        projected = torch.mm(flattened, Q[:, :min(hidden_dim//2, flattened.size(0))])
        # Pad back to original dimension
        if projected.size(1) < hidden_dim:
            padding = torch.zeros(projected.size(0), hidden_dim - projected.size(1), 
                                device=projected.device, dtype=projected.dtype)
            projected = torch.cat([projected, padding], dim=1)
        
        return projected.view(batch_size, seq_len, hidden_dim)
    except (RuntimeError, torch.linalg.LinAlgError):
        # Fallback to normalization
        return F.normalize(hidden_states, dim=-1)


def apply_transform(hidden_states: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the specified transform to hidden states"""
    if transform == "none":
        return hidden_states
    elif transform == "supr":
        return apply_suppressed_transform(hidden_states)
    elif transform == "ether":
        return apply_ether_transform(hidden_states) 
    elif transform == "ortho":
        return apply_ortho_transform(hidden_states)
    else:
        return hidden_states


class BaseDPOTrainer:
    """Base DPO trainer - handles the core preference optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        self.setup_data()
    
    @property 
    def uses_innerpo(self) -> bool:
        """Whether this trainer uses InnerPO components"""
        return self.config.method == "innerpo"
    
    def setup_model(self):
        """Load and setup model with LoRA"""
        print(f"Loading model: {self.config.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        
        # Add LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Reference model for DPO (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def setup_data(self):
        """Load dataset"""
        print(f"Loading dataset: {self.config.dataset}")
        
        # Load training data
        train_data = load_preference_dataset(
            self.config.dataset, 
            split="train", 
            n_samples=1000 if not hasattr(self.config, 'n_samples') else self.config.n_samples
        )
        
        print(f"Loaded {len(train_data)} training examples")
        if self.config.verbose > 1:
            print(f"Example: {train_data[0]}")
        
        dataset = SimpleDataset(train_data, self.tokenizer, self.config.max_length)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr
        )
        
        total_steps = len(self.dataloader) * 3  # Assume 3 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    
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
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        chosen_ids = batch["chosen_ids"].to(self.config.device)
        rejected_ids = batch["rejected_ids"].to(self.config.device)
        prompt_lens = batch["prompt_lens"].to(self.config.device)
        
        if self.config.method == "dpo":
            loss = self.dpo_loss(chosen_ids, rejected_ids, prompt_lens)
        else:  # InnerPO variants
            loss = self.innerpo_loss(chosen_ids, rejected_ids, prompt_lens)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training with method: {self.config.method}")
        
        if self.config.verbose > 0:
            wandb.init(
                project="innerpo-simplified",
                config=vars(self.config),
                name=f"{self.config.method}_{self.config.dataset}_{self.config.seed}"
            )
        
        step = 0
        for epoch in range(3):  # Simple 3 epoch training
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                loss = self.train_step(batch)
                
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                
                if self.config.verbose > 0:
                    wandb.log({"loss": loss, "step": step})
                
                step += 1
                if step >= self.config.max_steps:
                    break
            
            if step >= self.config.max_steps:
                break
        
        # Save model
        output_path = Path(self.config.output_dir) / f"{self.config.method}_{self.config.dataset}_seed{self.config.seed}"
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"Training complete. Model saved to {output_path}")


def main():
    # Use tyro for clean CLI with dataclass support
    config = tyro.cli(Config)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed) 
    torch.manual_seed(config.seed)
    
    # Train
    trainer = BaseDPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()