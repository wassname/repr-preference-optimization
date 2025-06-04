from typing import Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:

    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genies_preferences
    https://joshuaclymer.github.io/generalization-analogies-website/
    """
    lr: float = 7e-5
    weight_decay: float = 0.01


    dataset: str = "math"
    """train dataset."""

    # TODO manually set the ood and rnd datasets, or else hard code sets

    verbose: int = 1
    seed: int = 42

    dev: bool = False
    """fast run"""

    load_in_4bit: bool = False
    load_in_8bit: bool = True
    use_gradient_checkpointing: bool = False
    schedule: str = "wsd"

    n_samples: int = 20000
    eval_samples: Optional[int] = None
    max_length: int = 512
    max_prompt_length: int = 450 # on the math ds, prompts are 446 tokens long

    
    # # base_model: str = "wassname/llama-3-2-1b-sft"
    # batch_size: int = 5

    base_model: str = "wassname/Qwen3-0.6B-sft"
    batch_size: int = 7

    # # base_model: str = "wassname/SmolLM2-360M-sft"
    # batch_size: int = 10

    # base_model: str = "wassname/SmolLM2-135M-sft"
    # batch_size: int = 14

    save: bool = True
    wandb: bool = True

    _model_keys = []
    """when subclassing, add kwargs destined for the model to this list"""
