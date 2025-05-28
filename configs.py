"""
Configuration management for different models and hardware setups.

This provides presets for different model sizes and GPU configurations,
while still allowing override via CLI.
"""

import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
from train_lightning import Config


# Model-specific configurations
MODEL_CONFIGS = {
    # Tiny models for development
    "tiny": {
        "base_model": "reciprocate/tiny-llama",
        "batch_size": 8,
        "max_steps": 50,
        "lora_r": 8,
        "lora_alpha": 16,
        "layers": [2, 4, 6]
    },
    
    # Small models (1B params)
    "qwen-0.6b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "batch_size": 8,
        "max_steps": 500,
        "lora_r": 16,
        "lora_alpha": 32,
        "layers": [8, 12, 16]
    },
    
    "llama-1b": {
        "base_model": "unsloth/Llama-3.2-1B-Instruct",
        "batch_size": 6,
        "max_steps": 500,
        "lora_r": 16,
        "lora_alpha": 32,
        "layers": [8, 12, 16]
    },
    
    # Medium models (3-4B params)
    "qwen-4b": {
        "base_model": "Qwen/Qwen3-4B",
        "batch_size": 4,
        "max_steps": 500,
        "lora_r": 32,
        "lora_alpha": 64,
        "layers": [12, 18, 24]
    },
    
    "llama-3b": {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "batch_size": 4,
        "max_steps": 500,
        "lora_r": 32,
        "lora_alpha": 64,
        "layers": [12, 18, 24]
    },
    
    # Large models (7B+ params)
    "qwen-8b": {
        "base_model": "Qwen/Qwen3-8B",
        "batch_size": 2,
        "max_steps": 500,
        "lora_r": 64,
        "lora_alpha": 128,
        "layers": [16, 24, 32]
    },
    
    "qwen-14b": {
        "base_model": "Qwen/Qwen3-14B",
        "batch_size": 1,
        "max_steps": 500,
        "lora_r": 64,
        "lora_alpha": 128,
        "layers": [20, 30, 40]
    }
}

# GPU-specific configurations
GPU_CONFIGS = {
    # For development/testing
    "cpu": {
        "accelerator": "cpu",
        "devices": 1,
        "batch_size": 2,
        "max_steps": 10
    },
    
    # Single GPU setups
    "single-gpu": {
        "accelerator": "gpu",
        "devices": 1,
    },
    
    # High-memory GPU (A100, H100)
    "a100": {
        "accelerator": "gpu", 
        "devices": 1,
        "batch_size": 8,  # Can handle larger batches
    },
    
    # Multi-GPU setups
    "multi-gpu": {
        "accelerator": "gpu",
        "devices": "auto",  # Use all available
    }
}


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config_to_yaml(config: Config, yaml_path: str):
    """Save configuration to YAML file"""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def create_config(
    model_preset: Optional[str] = None,
    gpu_preset: Optional[str] = None,
    yaml_config: Optional[str] = None,
    **overrides
) -> Config:
    """
    Create a configuration by combining presets and overrides.
    
    Order of precedence (highest to lowest):
    1. CLI overrides (**overrides)
    2. YAML config file (yaml_config)
    3. Model preset (model_preset)  
    4. GPU preset (gpu_preset)
    5. Default Config values
    """
    
    # Start with default config
    config_dict = asdict(Config())
    
    # Apply GPU preset
    if gpu_preset and gpu_preset in GPU_CONFIGS:
        config_dict.update(GPU_CONFIGS[gpu_preset])
    
    # Apply model preset
    if model_preset and model_preset in MODEL_CONFIGS:
        config_dict.update(MODEL_CONFIGS[model_preset])
    
    # Apply YAML config
    if yaml_config:
        yaml_dict = load_config_from_yaml(yaml_config)
        config_dict.update(yaml_dict)
    
    # Apply CLI overrides
    config_dict.update(overrides)
    
    # Create and return config object
    return Config(**config_dict)


def list_presets():
    """Print available presets"""
    print("Available model presets:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config['base_model']} (batch_size={config['batch_size']})")
    
    print("\nAvailable GPU presets:")
    for name, config in GPU_CONFIGS.items():
        print(f"  {name}: {config}")


if __name__ == "__main__":
    # Demo usage
    list_presets()
    
    print("\n" + "="*50)
    print("Example configurations:")
    
    # Example 1: Small model on single GPU
    config1 = create_config(model_preset="qwen-0.6b", gpu_preset="single-gpu")
    print(f"\nSmall model: {config1.base_model}, batch_size={config1.batch_size}")
    
    # Example 2: Large model on A100
    config2 = create_config(model_preset="qwen-8b", gpu_preset="a100")
    print(f"Large model: {config2.base_model}, batch_size={config2.batch_size}")
    
    # Example 3: Custom overrides
    config3 = create_config(
        model_preset="llama-1b", 
        lr=5e-5,  # Custom learning rate
        max_steps=1000  # More steps
    )
    print(f"Custom: lr={config3.lr}, max_steps={config3.max_steps}")