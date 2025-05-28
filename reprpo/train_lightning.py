#!/usr/bin/env python3
"""
InnerPO with PyTorch Lightning - Clean and powerful

This version uses Lightning for better experiment management while keeping the same simplicity.
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
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import tyro

# Local imports  
from .data_loader import load_preference_dataset


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
    max_epochs: int = 3
    max_steps: int = 500
    max_length: int = 512
    eval_every_n_steps: int = 100
    
    # === LoRA ===
    lora_r: int = 16
    lora_alpha: int = 32
    
    # === Loss Parameters ===
    beta: float = 0.1           # DPO beta parameter
    innerpo_alpha: float = 0.1  # InnerPO loss weight (only for INNERPO method)
    
    # === InnerPO Layers ===
    layers: Optional[List[int]] = None  # Layers to apply InnerPO to
    
    # === System ===
    accelerator: str = "auto"
    devices: str = "auto"
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
            self.layers = [2, 4, 6]  # Default middle layers (adjusted for smaller models)
    
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
    
    # Handle different model structures (PEFT vs base model)
    base_model = model
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            base_model = model.base_model.model
        else:
            base_model = model.base_model
    elif hasattr(model, 'model'):
        base_model = model.model
    
    # Get the layers (different names for different architectures)
    model_layers = None
    
    # For LlamaForCausalLM, the layers are in model.layers
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        model_layers = base_model.model.layers
    elif hasattr(base_model, 'layers'):
        model_layers = base_model.layers
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        model_layers = base_model.transformer.h
    elif hasattr(base_model, 'h'):
        model_layers = base_model.h
    
    if model_layers is None:
        print(f"Warning: Could not find layers in model structure. Model type: {type(base_model)}")
        print(f"Available attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')]}")
        return hidden_states
    
    for layer_idx in layers:
        if layer_idx < len(model_layers):
            hook = model_layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
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


class InnerPOLightningModule(pl.LightningModule):
    """Lightning module for InnerPO training"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))
        
        # Setup model
        self.setup_model()
        
    def setup_model(self):
        """Load and setup model with LoRA"""
        print(f"Loading model: {self.config.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map=None  # Let Lightning handle device placement
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
            torch_dtype=torch.float16,
            device_map=None
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        prompt_lens = batch["prompt_lens"]
        
        # Always compute DPO loss for comparison
        dpo_loss_val = dpo_loss(self.model, self.ref_model, chosen_ids, rejected_ids, 
                               prompt_lens, self.config.beta)
        
        # Choose loss function based on method
        if self.config.method == "dpo":
            total_loss = dpo_loss_val
            innerpo_loss_val = torch.tensor(0.0, device=self.device)
        elif self.config.method == "innerpo":
            # For InnerPO, we want to track both components
            innerpo_loss_val = innerpo_loss(
                self.model, self.ref_model, chosen_ids, rejected_ids,
                prompt_lens, self.config.layers, self.config.transform,
                self.config.beta, self.config.innerpo_alpha
            )
            total_loss = innerpo_loss_val
            # Subtract DPO component to get pure InnerPO component
            innerpo_component = innerpo_loss_val - dpo_loss_val
        elif self.config.method == "projgrad":
            total_loss = projgrad_loss(self.model, self.ref_model, chosen_ids, rejected_ids,
                                     prompt_lens, self.config.layers, self.config.beta)
            innerpo_loss_val = torch.tensor(0.0, device=self.device)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Log only essential metrics
        metrics = {
            "train_loss": total_loss,
            "dpo_loss": dpo_loss_val,
        }
        
        # Add InnerPO-specific metrics if relevant
        if self.config.method == "innerpo":
            metrics["innerpo_component"] = innerpo_component
            
        for name, value in metrics.items():
            self.log(name, value, on_step=True, on_epoch=True, prog_bar=(name == "train_loss"))
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - same as training but without gradients"""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        
        # Calculate total steps for scheduler
        # This is approximate since we don't know exact dataset size here
        total_steps = self.config.max_steps
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


class InnerPODataModule(pl.LightningDataModule):
    """Lightning data module for preference datasets"""
    
    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
    
    def setup(self, stage: str = None):
        """Setup train/val datasets"""
        # Load training data
        train_data = load_preference_dataset(
            self.config.dataset, 
            split="train", 
            n_samples=1000
        )
        
        # Split for validation (80/20)
        split_idx = int(0.8 * len(train_data))
        self.train_data = train_data[:split_idx]
        self.val_data = train_data[split_idx:]
        
        print(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val examples")
    
    def train_dataloader(self):
        dataset = SimpleDataset(self.train_data, self.tokenizer, self.config.max_length)
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2
        )
    
    def val_dataloader(self):
        dataset = SimpleDataset(self.val_data, self.tokenizer, self.config.max_length)
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )


def main():
    # Use tyro for clean CLI with dataclass support
    config = tyro.cli(Config)
    
    # Set random seeds
    pl.seed_everything(config.seed)
    
    # Create Lightning module
    model = InnerPOLightningModule(config)
    
    # Create data module
    data_module = InnerPODataModule(config, model.tokenizer)
    
    # Setup logging
    logger = None
    if config.verbose > 0:
        logger = WandbLogger(
            project="innerpo-lightning",
            name=f"{config.method_name}_{config.dataset}_seed{config.seed}",
            config=vars(config)
        )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.output_dir) / config.method_name,
        filename=f"{config.dataset}_seed{config.seed}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    callbacks.append(checkpoint_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=config.eval_every_n_steps,
        log_every_n_steps=10,
        precision="16-mixed",  # Use mixed precision for speed
        enable_progress_bar=config.verbose > 0
    )
    
    # Train
    trainer.fit(model, data_module)
    
    print(f"Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()