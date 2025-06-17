from typing import Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:

    lr: float = 2e-5
    """1e-6 in alignment handbook with cosine, 5e-7 is eric mitchells ref, SimPPo is 6e-7 or lower"""

    weight_decay: float = 0.0
    """0 in alignment handbook, and eric mitchells ref"""

    gradient_clip_val: float = 10.0

    ideal_batch_size: int = 16
    """ideal batch size, used to calculate gradient accumulation steps 16-128 are used in ref repos"""

    pl_precision: str = "bf16"
    """precision for pytorch lightning, bf16-mixed is best for 8B models on 80GB GPUs. 'transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16'. See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer"""

    num_workers: int = 8
    """number of workers for dataloader, 0 is best for 80GB GPUs"""

    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genies_preferences
    https://joshuaclymer.github.io/generalization-analogies-website/
    """
    dataset: str = "us_history"
    # dataset: str = "HuggingFaceH4/ultrafeedback_binarized:train_prefs"
    """train dataset."""

    verbose: int = 1
    seed: int = 42
    patience: int = 3

    dev: bool = False
    """fast run. Seems to greatly harm performance"""

    load_in_4bit: bool = False
    """bit base and adam 8bit opt. Seems to greatly harm performance"""

    load_in_8bit: bool = False
    """bnb 8bit, and adam 8bit opt"""

    use_grad_paging: bool = False
    """avoid mem spikes"""

    n_samples: int = 10000 # 3k in dataset, most ref models for 1 epoch of 60k ultrafeedback samples
    eval_samples: Optional[int] = 300
    max_length: int = 512
    max_prompt_length: int = 450 # on the math ds, prompts are 446 tokens long

    # # 80GB gpu
    # base_model: str = "princeton-nlp/Llama-3-Base-8B-SFT"
    # batch_size: int = 7

    # allenai/Llama-3.1-Tulu-3-8B-SFT

    # base_model: str = "wassname/llama-3.2-3b-sft"
    # batch_size: int = 10

    # base_model: str = "allenai/OLMo-2-0425-1B-SFT"
    # batch_size: int = 20
    
    # 24GB gpu
    
    # # base_model: str = "allenai/OLMo-2-0425-1B-SFT"
    # batch_size: int = 2

    base_model: str = "wassname/Qwen3-0.6B-sft"
    batch_size: int = 4

    # # base_model: str = "wassname/SmolLM2-360M-sft"
    # batch_size: int = 7

    # base_model: str = "wassname/SmolLM2-135M-sft"
    # batch_size: int = 12

    save: bool = True
    wandb: bool = True

    _model_keys = []
    """when subclassing, add kwargs destined for the model to this list"""
