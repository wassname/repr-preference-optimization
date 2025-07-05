"""
This file is so we can run a bunch of unit tests
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["REPR_CONFIG"] = "../configs/dev.yaml"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "offline"

import tyro
import yaml
from reprpo.training import train, apply_cfg_overrides_from_env_var
from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.experiments import experiment_configs
from pathlib import Path
import pytest

from reprpo import silence
# silence.test()

args_list = [
    "dpo --use-mallows",
    "hs-none-topk --calc-mallows --loss.use-mallows",
    "hs-ether-InnerDPO --calc-mallows --loss.use-mallows",
    "hs-supr-topk --calc-wpo --loss.use-policy-weights",
    "hs-none-topk --loss.use-token-constraint",
    "hs-none-mse --loss.use-token-constraint",
    "hs-none-rank --loss.use-token-constraint",
    "dpo_projgrad",
    "dpo_projbp",
]

@pytest.mark.parametrize("args", args_list)
def test_train_experiments_dev(args):
    args = args.split()
    training_args = tyro.extras.overridable_config_cli(configs=experiment_configs, args=args)
    training_args = apply_cfg_overrides_from_env_var(training_args)
    train(training_args)



if __name__ == "__main__":
    for args in args_list:
        args = args.split()
        training_args = tyro.extras.overridable_config_cli(configs=experiment_configs, args=args)
        training_args = apply_cfg_overrides_from_env_var(training_args)
        train(training_args)
