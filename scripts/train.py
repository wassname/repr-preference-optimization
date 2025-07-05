import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tyro
import yaml
from reprpo.training import train, apply_cfg_overrides_from_env_var
from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.experiments import experiment_configs
from pathlib import Path



    

if __name__ == "__main__":
    training_args = tyro.extras.overridable_config_cli(experiment_configs, use_underscores=True)

    # tyro has a default option, but it doesn't work with subcommands, so I apply overides manually
    # TODO should really put cli arg after this
    training_args = apply_cfg_overrides_from_env_var(training_args)

    train(training_args)
