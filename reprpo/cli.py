#!/usr/bin/env python3
"""
Clean hierarchical CLI for InnerPO using tyro.

This gives you a nice structure like:
    python -m reprpo.cli train dpo --model.base_model=Qwen/Qwen3-0.6B --data.dataset=math
    python -m reprpo.cli train innerpo --method.transform=supr --model.lora_r=32
    python -m reprpo.cli eval ./outputs/model --eval.datasets=[math,code]
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal, Union, Tuple
from enum import Enum

import torch
import numpy as np
import tyro
import lightning.pytorch as pl

# Local imports
from .train_lightning import InnerPOLightningModule, InnerPODataModule
from .evaluate import paper_eval, eval_with_open_pref_eval


class Transform(Enum):
    """Hidden state transforms"""
    NONE = "none"
    SUPR = "supr"      # Suppressed 
    ETHER = "ether"    # ETHER
    ORTHO = "ortho"    # Orthogonal


class Dataset(Enum):
    """Available datasets"""
    MATH = "math"
    CODE = "code" 
    ALPACA_MMLU = "alpaca_mmlu"
    COOKING = "cooking"
    ALPACA_LOW_QUALITY = "alpaca_low_quality"
    MATHS_EASY = "maths_easy"


@dataclass
class ModelConfig:
    """Model configuration"""
    base_model: str = "Qwen/Qwen3-0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    layers: Tuple[int, ...] = (8, 12, 16)  # Use tuple instead of list for tyro compatibility


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: Dataset = Dataset.MATH
    batch_size: int = 4
    max_length: int = 512


@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float = 1e-5
    max_epochs: int = 3
    max_steps: int = 500
    beta: float = 0.1          # DPO beta
    innerpo_alpha: float = 0.1 # InnerPO weight
    seed: int = 1


@dataclass 
class SystemConfig:
    """System configuration"""
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed"
    output_dir: str = "./outputs"
    verbose: int = 1


@dataclass
class DPOMethod:
    """Standard DPO training"""
    pass


@dataclass  
class InnerPOMethod:
    """InnerPO training with hidden state alignment"""
    transform: Transform = Transform.SUPR


@dataclass
class ProjGradMethod:
    """Projection gradient method"""
    pass


# Union of all methods
Method = Union[DPOMethod, InnerPOMethod, ProjGradMethod]


@dataclass
class TrainCommand:
    """Train a model using the specified method"""
    method: Method
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    system: SystemConfig = SystemConfig()
    
    def run(self):
        """Execute training"""
        # Set seeds
        pl.seed_everything(self.training.seed)
        
        # Convert to flat config for Lightning module
        config = self._to_lightning_config()
        
        # Create Lightning components
        model = InnerPOLightningModule(config)
        data_module = InnerPODataModule(config, model.tokenizer)
        
        # Setup logging
        logger = None
        if self.system.verbose > 0:
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(
                project="innerpo",
                name=f"{config.method_name}_{config.dataset}_seed{config.seed}",
                config=vars(config)
            )
        
        # Setup callbacks
        callbacks = []
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.system.output_dir) / config.method_name,
            filename=f"{config.dataset}_seed{config.seed}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_callback)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.training.max_epochs,
            max_steps=self.training.max_steps,
            accelerator=self.system.accelerator,
            devices=self.system.devices,
            logger=logger,
            callbacks=callbacks,
            precision=self.system.precision,
            enable_progress_bar=self.system.verbose > 0,
            log_every_n_steps=10,
        )
        
        # Train
        trainer.fit(model, data_module)
        print(f"Training complete! Model saved to: {checkpoint_callback.best_model_path}")
    
    def _to_lightning_config(self):
        """Convert hierarchical config to flat config for Lightning"""
        from .train_lightning import Config
        
        # Determine method details
        if isinstance(self.method, DPOMethod):
            method = "dpo"
            transform = "none"
        elif isinstance(self.method, InnerPOMethod):
            method = "innerpo" 
            transform = self.method.transform.value
        elif isinstance(self.method, ProjGradMethod):
            method = "projgrad"
            transform = "none"
        else:
            raise ValueError(f"Unknown method: {type(self.method)}")
        
        return Config(
            method=method,
            transform=transform,
            base_model=self.model.base_model,
            dataset=self.data.dataset.value,
            seed=self.training.seed,
            lr=self.training.lr,
            batch_size=self.data.batch_size,
            max_epochs=self.training.max_epochs,
            max_steps=self.training.max_steps,
            max_length=self.data.max_length,
            lora_r=self.model.lora_r,
            lora_alpha=self.model.lora_alpha,
            beta=self.training.beta,
            innerpo_alpha=self.training.innerpo_alpha,
            layers=list(self.model.layers),
            accelerator=self.system.accelerator,
            devices=self.system.devices,
            output_dir=self.system.output_dir,
            verbose=self.system.verbose,
        )


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    ref_model: str = "Qwen/Qwen3-0.6B"
    train_dataset: str = "math"  # What dataset was the model trained on
    paper_mode: bool = True      # Use comprehensive evaluation for paper


@dataclass
class EvalCommand:
    """Evaluate a trained model with comprehensive distribution shift analysis"""
    model_path: str
    eval: EvalConfig = EvalConfig()
    
    def run(self):
        """Execute evaluation using open_pref_eval for distribution shift analysis"""
        if self.eval.paper_mode:
            # Comprehensive evaluation for paper
            results = paper_eval(
                self.model_path, 
                self.eval.ref_model, 
                self.eval.train_dataset
            )
        else:
            # Basic evaluation
            results = eval_with_open_pref_eval(
                self.model_path,
                self.eval.ref_model, 
                self.eval.train_dataset
            )
            
            print(f"\nEvaluation Results for {self.model_path}")
            print("=" * 50)
            for dataset, accuracy in results.items():
                print(f"{dataset:15s}: {accuracy:.3f}")
        
        return results


@dataclass
class ConfigCommand:
    """Show example configurations"""
    
    def run(self):
        """Show examples"""
        print("InnerPO CLI Examples:")
        print("\n1. Basic DPO training:")
        print("   python -m reprpo.cli train dpo")
        
        print("\n2. InnerPO with SUPR transform:")
        print("   python -m reprpo.cli train innerpo --method.transform=supr")
        
        print("\n3. Custom model and data:")
        print("   python -m reprpo.cli train innerpo \\")
        print("     --method.transform=ether \\") 
        print("     --model.base_model=unsloth/Llama-3.2-1B-Instruct \\")
        print("     --data.dataset=code \\")
        print("     --data.batch_size=8")
        
        print("\n4. Evaluation:")
        print("   python -m reprpo.cli eval ./outputs/innerpo-supr_math_seed1")
        
        print("\n5. Dev mode (quick test):")
        print("   python -m reprpo.cli train dpo \\")
        print("     --model.base_model=reciprocate/tiny-llama \\")
        print("     --training.max_steps=50 \\") 
        print("     --system.verbose=2")


# Main CLI interface
def main():
    tyro.cli({
        "train": TrainCommand,
        "eval": EvalCommand, 
        "config": ConfigCommand,
    }).run()


if __name__ == "__main__":
    main()