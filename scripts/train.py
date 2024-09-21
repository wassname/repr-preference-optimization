import os
import tyro
import yaml
from reprpo.training import train
from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.experiments import experiment_configs

if __name__ == "__main__":
    training_args = tyro.extras.overridable_config_cli(experiment_configs)

    # tyro has a default option, but it doesn't work with subcommands, so I apply overides manually
    # e.g. REPR_CONFIG=./configs/dev.yaml
    overrides = {}
    f = os.environ.get("REPR_CONFIG")
    print("f", f)
    if f is not None:
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            if hasattr(training_args, k):
                setattr(training_args, k, v)
            else:
                print(f"Warning: {k} not found in training_args")
        print(f"loaded default config from {f}")
        # print(args)

    train(training_args)
