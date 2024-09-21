import os
import tyro
import yaml
from reprpo.interventions.config import ExperimentConfig
from reprpo.training import train
from reprpo.interventions import Interventions
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms

default_configs = {

    "reprpo": (
        "Small experiment.",
        ExperimentConfig(
            intervention=Interventions.reprpo.value
        ),
    ),
    "dpo": (
        "DPO experiment.",
        ExperimentConfig(
            intervention=Interventions.dpo.value(
            ),
        ),
    ),
}

if __name__ == "__main__":
    training_args = tyro.extras.overridable_config_cli(default_configs)

    training_args = tyro.cli(ExperimentConfig)

    # tyro has a default option, but it doesn't work with subcommands, so I apply overides manually
    # e.g. REPR_CONFIG=./configs/dev.yaml
    overrides = {}
    f = os.environ.get("REPR_CONFIG")
    print("f", f)
    if f is not None:
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            setattr(training_args, k, v)
        print(f"loaded default config from {f}")
        # print(args)

    train(training_args)
