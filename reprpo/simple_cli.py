#!/usr/bin/env python3
"""
Simplified working CLI for InnerPO - focuses on core functionality
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("InnerPO CLI Examples:")
        print("\n1. Basic DPO training:")
        print("   python -m reprpo.simple_cli train dpo")
        
        print("\n2. InnerPO with SUPR transform:")
        print("   python -m reprpo.simple_cli train innerpo supr")
        
        print("\n3. InnerPO with ETHER transform:")
        print("   python -m reprpo.simple_cli train innerpo ether")
        
        print("\n4. Evaluation:")
        print("   python -m reprpo.simple_cli eval ./outputs/model_path")
        
        print("\n5. Dev mode (quick test):")
        print("   python -m reprpo.simple_cli train dpo --dev")
        return
    
    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Usage: python -m reprpo.simple_cli train <method> [transform] [--dev]")
            print("Methods: dpo, innerpo, projgrad")
            print("Transforms: none, supr, ether, ortho")
            return
            
        method = sys.argv[2]
        transform = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else "none"
        dev_mode = "--dev" in sys.argv
        
        print(f"Training with method: {method}, transform: {transform}")
        if dev_mode:
            print("Dev mode: using tiny model and 50 steps")
            
        # Import and run training
        from .train_lightning import Config, InnerPOLightningModule, InnerPODataModule
        import lightning.pytorch as pl
        
        # Create config
        config = Config(
            method=method,
            transform=transform,
            base_model="reciprocate/tiny-llama" if dev_mode else "Qwen/Qwen3-0.6B",
            dataset="math",
            seed=1,
            max_steps=50 if dev_mode else 500,
            batch_size=2 if dev_mode else 4,
            verbose=1
        )
        
        # Set seeds
        pl.seed_everything(config.seed)
        
        # Create Lightning components
        model = InnerPOLightningModule(config)
        data_module = InnerPODataModule(config, model.tokenizer)
        
        # Create trainer (disable default checkpointing to avoid litmodels suggestions)
        trainer = pl.Trainer(
            max_steps=config.max_steps,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10,
            precision="16-mixed" if not dev_mode else "32",
            enable_checkpointing=False,  # Disable auto-checkpointing, we'll save manually
        )
        
        # Train
        print("Starting training...")
        trainer.fit(model, data_module)
        
        # Save model
        output_path = Path("./outputs") / f"{config.method_name}_{config.dataset}_seed{config.seed}"
        output_path.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        print(f"Training complete! Model saved to: {output_path}")
        
    elif command == "eval":
        if len(sys.argv) < 3:
            print("Usage: python -m reprpo.simple_cli eval <model_path>")
            return
            
        model_path = sys.argv[2]
        print(f"Evaluating model: {model_path}")
        
        # Import and run evaluation
        from .evaluate import quick_eval
        
        results = quick_eval(model_path, "Qwen/Qwen3-0.6B")
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS:")
        for dataset, accuracy in results.items():
            print(f"{dataset:15s}: {accuracy:.3f}")
            
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval")


if __name__ == "__main__":
    main()