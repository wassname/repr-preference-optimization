import os
import tyro
import yaml
from reprpo.interventions import Interventions, InterventionType
from reprpo.training import train

if __name__ == "__main__":
    training_args = tyro.cli(InterventionType)

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
