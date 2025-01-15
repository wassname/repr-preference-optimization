import os
import tyro
import yaml
from reprpo.training import train, apply_cfg_overrides
from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.experiments import experiment_configs
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



    

if __name__ == "__main__":
    training_args = tyro.extras.overridable_config_cli(experiment_configs)

    # tyro has a default option, but it doesn't work with subcommands, so I apply overides manually
    # TODO should really put cli arg after this
    training_args = apply_cfg_overrides(training_args)

    train(training_args)
