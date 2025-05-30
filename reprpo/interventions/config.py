from typing import Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:

    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genies_preferences
    https://joshuaclymer.github.io/generalization-analogies-website/
    """


    dataset: str = "math"
    """train dataset."""

    # TODO manually set the ood and rnd datasets, or else hard code sets

    verbose: int = 1

    dev: bool = False
    """fast run"""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = False

    batch_size: int = 10

    n_samples: int = 10000
    eval_samples: Optional[int] = None
    max_length: int = 256
    max_prompt_length: int = 128

    
    base_model: str = "wassname/llama-3-2-1b-sft"
    base_model: str = "Qwen/Qwen3-0.6B"

    save: bool = True
    wandb: bool = True
