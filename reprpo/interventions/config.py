import torch
from typing import Optional, Callable, Union
from dataclasses import dataclass



@dataclass
class ExperimentConfig:
    
    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genie_dpo"""

    dataset: str = 'us_history_textbook'
    """train dataset."""

    verbose: bool = False

    dev: bool = False
    """fast run"""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = False

    batch_size: int = 16

    n_samples: int = 1800 * 3
    eval_samples: Optional[int] = None
    max_length: int = 196
    max_prompt_length: int = 96
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    



