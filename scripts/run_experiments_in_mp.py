"""run over all experimental configs."""
import os
import gc
from loguru import logger
import pandas as pd
from reprpo.experiments import experiment_configs
from reprpo.training import train, apply_cfg_overrides
from open_pref_eval.datasets import ds2name, load_dataset_n

import psutil
import datasets
total_ram = psutil.virtual_memory().total
datasets.config.IN_MEMORY_MAX_SIZE = int(total_ram * 0.3)
print(f"IN_MEMORY_MAX_SIZE set to {datasets.config.IN_MEMORY_MAX_SIZE} bytes")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
timestamp = pd.Timestamp.now('Australia/Perth').strftime("%d%b%H%M") # e.g. '11Oct0054', works for me
os.environ["WANDB_GROUP"] = f"exp-{timestamp}"
keys = list(experiment_configs.keys())
print(" ".join(keys))  # Print keys as a comma-separated string


# for i, (name, (_, training_args)) in enumerate(experiment_configs.items()):
    
import multiprocessing as mp
import sys
from open_pref_eval.datasets.genies import GENIES
datasets = [r['source'] for r in GENIES]
# pre load datasets
for ds in datasets:
    print(ds)
    load_dataset_n('wassname/genies_preferences', name=ds, split='test', N=1000)

def train_wrapper(params):
    import torch
    sys.stdout = sys.__stdout__  # Restore stdout
    train(params)

def run_experiment(params):
    p = mp.Process(target=train_wrapper, args=(params,))
    p.start()
    p.join()
    gc.collect()

for i, (name, (_, training_args)) in enumerate(experiment_configs.items()):
    print(f"Running experiment {i} {name}")
    training_args = apply_cfg_overrides(training_args)
    if i == 0:
        training_args.verbose = 3
    try:
        run_experiment(training_args)
    # except torch.OutOfMemoryError as e:
    #     logger.error(f"OOM error in training {training_args}")
    except KeyboardInterrupt:
        logger.error(f"KeyboardInterrupt in training {training_args}")
        break
    except Exception as e:
        logger.exception(f"Error {e} in training {training_args}")

main_experiments = list(experiment_configs.items())[:4]
for i, (name, (_, training_args)) in enumerate(main_experiments):
    for ds in datasets:
        print(f"Running experiment {i} {name} on {ds}")
        training_args = apply_cfg_overrides(training_args)
        training_args.dataset = ds
        if i == 0:
            training_args.verbose = 3
        try:
            run_experiment(training_args)
        # except torch.OutOfMemoryError as e:
        #     logger.error(f"OOM error in training {training_args}")
        except KeyboardInterrupt:
            logger.error(f"KeyboardInterrupt in training {training_args}")
            break
        except Exception as e:
            logger.exception(f"Error {e} in training {training_args}")
        
