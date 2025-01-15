"""run over all experimental configs."""
from reprpo.experiments import experiment_configs
import torch
from reprpo.training import train, apply_cfg_overrides
import os
import gc
from loguru import logger
from datetime import datetime
import pandas as pd
from reprpo.helpers.torch import clear_mem
from open_pref_eval.datasets import ds2name, load_dataset_n

import psutil
import datasets
total_ram = psutil.virtual_memory().total
datasets.config.IN_MEMORY_MAX_SIZE = int(total_ram * 0.3)
print(f"IN_MEMORY_MAX_SIZE set to {datasets.config.IN_MEMORY_MAX_SIZE} bytes")

# https://pytorch.org/docs/stable/notes/cuda.html#memory-management
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# garbage_collection_threshold
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="expandable_segments:True" # seems to stop gpu mem from filling up despite clearing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
timestamp = pd.Timestamp.now('Australia/Perth').strftime("%d%b%H%M") # e.g. '11Oct0054', works for me
os.environ["WANDB_GROUP"] = f"exp-{timestamp}"
keys = list(experiment_configs.keys())
print(" ".join(keys))  # Print keys as a comma-separated string

# shuffle keys
# import random
# random.shuffle(keys)
for i, name in enumerate(keys):
    training_args = experiment_configs[name][1]
    training_args = apply_cfg_overrides(training_args)
    print(f"Running experiment {i} {name}")

    if i == 0:
        training_args.verbose = 3
    # else:
    #     os.environ['HF_DATASETS_OFFLINE ']= “1”
    try:
        train(training_args)
    except torch.OutOfMemoryError as e:
        logger.error(f"OOM error in training {training_args}")
    except KeyboardInterrupt:
        logger.error(f"KeyboardInterrupt in training {training_args}")
        break
    except Exception as e:
        logger.exception(f"Error {e} in training {training_args}")
    clear_mem()


from open_pref_eval.datasets.genies import GENIES
datasets = [r['source'] for r in GENIES]
# pre load datasets
for ds in datasets:
    print(ds)
    load_dataset_n('wassname/genies_preferences', name=ds, split='test', N=1000)

# run on other ds's
main_experiments = list(experiment_configs.items())[:4]
for ds in datasets:
    os.environ["WANDB_GROUP"] = f"exp-{timestamp}-{ds}"
    for i, (name, (_, training_args)) in enumerate(main_experiments):
        print(f"Running experiment {i} {name} on {ds}")
        training_args = apply_cfg_overrides(training_args)
        training_args.dataset = ds
        if i == 0:
            training_args.verbose = 3
        try:
            train(training_args)
        except torch.OutOfMemoryError as e:
            logger.error(f"OOM error in training {training_args}")
        except KeyboardInterrupt:
            logger.error(f"KeyboardInterrupt in training {training_args}")
            break
        except Exception as e:
            logger.exception(f"Error {e} in training {training_args}")
        
        clear_mem()
