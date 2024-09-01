# %% [markdown]
# Instead of using the complex TRL we code it from scratch, using lighting
# 
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

# %%
from pathlib import Path

# ML
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int, Bool
from torch.utils.data import DataLoader

# Numeric
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# lightning
import lightning as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger


# %%
# Local
from reprpo.helpers.torch import clear_mem
from reprpo.gen import generation_test
import reprpo.silence
from reprpo.helpers.lightning_hist import read_metrics_csv, plot_hist

from reprpo.data.collate import DPODataCollatorWithPadding
from reprpo.train.dpo import compute_dpo_loss_batch, PL_DPO_MODEL

# %%
torch.set_float32_matmul_precision("high")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from reprpo.helpers.wandb import init_wandb
nb_name = init_wandb(__file__)

# %%
from simple_parsing import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--method', type=str, default='dpo', help='dpo, reprpo_svd, reprpo_side')
parser.add_argument('--dataset', type=str, default='code_easy', help='code_easy etc see subsets in https://huggingface.co/datasets/wassname/genie_dpo')
parser.add_argument('--verbose', type=bool, default=False, help='print dataset')
args1, _ = parser.parse_known_args()


if args1.method == 'dpo':
    from reprpo.train.dpo import DPOTrainingArguments as TrainingArguments, PL_DPO_MODEL as PL_MODEL
elif args1.method == 'reprpo_svd':
    from reprpo.train.reprpo_svd import ReprPOSVDTrainingArguments as TrainingArguments, PL_REPRPO_SVD_MODEL as PL_MODEL
elif args1.method == 'reprpo_side':
    from reprpo.train.reprpo_side import ReprPOSideTrainingArguments as TrainingArguments, PL_REPRPO_SIDE_MODEL as PL_MODEL
else:
    raise ValueError(f"method {args1.method} not found")

parser.add_arguments(TrainingArguments, dest='args')
args2 = parser.parse_args()
args = TrainingArguments(**args2.args.__dict__)

# %% [markdown]
# ## Load model

from peft import LoraConfig, get_peft_model
from reprpo.models.load import load_model, print_trainable_parameters


model, tokenizer = load_model(args.model_name, load_in_4bit=args.load_in_4bit,  
                              attn_implementation='eager' # for gemma
)

# %% [markdown]
# ### Load adapter

# %%
from peft.tuners import BOFTConfig, OFTConfig, LoraConfig, IA3Config


adapter_name = f"{args.adapter_name}-{args1.dataset}"

peft_config = LoraConfig(
    # lora_alpha=16, 
    # r=16,
    # use_rslora=True,
    # use_dora=True,
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "v_proj"], # gemma, llama
    # target_modules=[
        # FIXME: I'm not sure we can do LORA on the layer we are targeting?
        # "qkv_proj", "gate_up_proj", # in
        # "down_proj",  "o_proj", # out
        #             ], # PHI3
)
model = get_peft_model(model, peft_config, adapter_name=adapter_name)
print_trainable_parameters(model)
if args1.verbose:
    print(model)

# %% [markdown]
# ## Load data

# %%
from datasets import load_dataset
dataset2 = load_dataset("wassname/genie_dpo", name=args1.dataset)

if args1.verbose:
    print(dataset2)

    print('QC one dataset row')
    r = dataset2['train'][0]
    print(r['prompt'])
    print('===')
    print(r['chosen'])
    print('---')
    print(r['rejected'])

# %% [markdown]
# ### Data Loader
# 
# We use huggingface datasets, which are pretokenized. So that we can stack

# %%
def tokenize_row(feature, tokenizer, args: TrainingArguments):
    """
    Tokenize a single row from a DPO specific dataset.

    see https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L784
    """
    batch = {}
    batch["chosen"] = tokenizer(feature["chosen"])["input_ids"]
    batch["rejected"] = tokenizer(feature["rejected"])["input_ids"]
    batch["prompt"] = tokenizer(feature["prompt"])["input_ids"]
    return batch

# %%
dataset3 = dataset2.map(lambda x: tokenize_row(x, tokenizer, args), batched=True, writer_batch_size=10)


# %%
custom_collate_fn = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id, 
                                                  tokenizer=tokenizer,
                                                  max_length=args.max_length,
                                                  mask_prompt_tokens=True,
                                                  max_prompt_length=args.max_prompt_length,
                                                  #label_pad_token_id=-100
                                                  )



# %%


ds = dataset3
dl_train = DataLoader(ds['train'], batch_size=args.batch_size, collate_fn=custom_collate_fn)

dl_val = DataLoader(ds['test'], batch_size=args.batch_size, collate_fn=custom_collate_fn)

if args1.verbose:
    print('QC one train batch')
    batch = next(iter(dl_train))
    batch.keys()

# %% [markdown]
# ## Trainer

# %% [markdown]
# - https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
# - https://gist.github.com/wassname/e29d02b5026a531e13912cf768e6fdc8

# %%
max_steps = args.n_samples // args.batch_size
print('max optimiser steps', max_steps)

# %%
ideal_batch_size = max(16, args.batch_size) # probobly wont be stable with less than 16, so make up the difference with gradient accumulation
accumulate_grad_batches = np.ceil(ideal_batch_size/args.batch_size).astype(int)
print('accumulate_grad_batches', accumulate_grad_batches)
print('accumulated batch size', args.batch_size*accumulate_grad_batches)

print(f"epochs {args.n_samples//len(dl_train)}")

# %%
from lightning.pytorch.callbacks import LearningRateMonitor
from reprpo.train.lightning import GenCallback

# %%
pl_model = PL_DPO_MODEL(model,
                 weight_decay=args.weight_decay,
                lr=args.lr,
                num_iterations=max_steps,
                batch_size=args.batch_size,
                )


# %%
# from reprpo.helpers.lightning_save import AdapterModelCheckpoint

# checkpoint_callback = AdapterModelCheckpoint(
#     verbose=True,
# )

# %%
from reprpo.helpers.lightning_existing_bnb import ExistingBitsandbytesPrecision

from accelerate.utils import CustomDtype
precision = ExistingBitsandbytesPrecision(
    # dtype=torch.bfloat16,
    # dtype=CustomDtype.INT4,
    dtype=torch.int8,
    default_dtype=torch.bfloat16,
)

# %%

model_fname = f'{args.model_name.replace("/","")}/{args.adapter_name}/{args1.dataset}'

timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"../outputs/{model_fname}/{timestamp}"
Path(save_dir).mkdir(exist_ok=True, parents=True)

callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # checkpoint_callback
        ]
if args1.verbose:
    callbacks+=[GenCallback(every=max_steps//4)]

trainer = pl.Trainer(
        max_steps=max_steps,
#         limit_val_batches=max_batches//5,
        gradient_clip_val=20,

        # accelerator='gpu',
        devices=1, 
        plugins=precision if args.load_in_4bit else "bf16",
        
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html
        # precision="bf16-true", # "32-true" "transformer-enginer
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=[
            CSVLogger(name=nb_name, save_dir=save_dir, flush_logs_every_n_steps=5),
            WandbLogger(name=nb_name, save_dir=save_dir),
        ],
        default_root_dir=save_dir,

        # too large, we will just save adapter afterwards
        enable_checkpointing=False, 

        # fast_dev_run=True,
    )

# train
trainer.fit(pl_model, dl_train, dl_val)

# %%
# save as regular adapter only

model.save_pretrained(
    save_dir+'-adapter',
)
print(f'saved to {save_dir}-adapter')


# %% [markdown]
# ### Hist

# %%


df_hist = read_metrics_csv(trainer.logger.experiment.metrics_file_path).bfill().ffill()
print(df_hist)

# import matplotlib
# plt.style.use('ggplot')
# matplotlib.rcParams['figure.figsize'] = (6, 2)
# plot_hist(df_hist, ['.*/loss_step', '.*/acc.*', '.*/auroc.*', '.*/.*reward_step'])


# %% [markdown]
# ## Gen

# %%
model.cuda(); # for some reason it ends up cpu

# %%
from reprpo.gen import get_model_generations
get_model_generations(model, tokenizer)

# %% [markdown]
# ## Eval

# %%
from reprpo.helpers.shypothesis import shypothesis
from reprpo.evaluate import evaluate_adapters
from open_pref_eval.evaluation import evaluate_model
from open_pref_eval.plot.radar import radar_plot
from open_pref_eval.datasets.genies import dist2datasets, GENIES
from open_pref_eval.datasets.ethics import get_ethics_datasets

# eval on ethics, GENIES, and our train dataset
N = None
datasets = [dataset2]
datasets += dist2datasets(GENIES, N=N)
datasets += get_ethics_datasets(N=N)


clear_mem()
res, df_res2 = evaluate_model(model=model, 
                              tokenizer=tokenizer, 
                              datasets=datasets,
                                 batch_size=args2.batch_size//2,
                                 bf16=True,
                                 torch_empty_cache_steps=100,)


# %%
from open_pref_eval.plot.radar import radar_plot
df_res = df_res2.groupby(['dataset', 'adapter'], dropna=False)['correct'].mean().unstack()
print(res)

# save
f = save_dir+'/eval.parquet'
df_res.to_parquet(f)
print(f'saved results to {f}')

radar_plot(df_res)
df_res

# %%
# print acc for journal
c  = df_res2.groupby(['adapter', 'dataset']).count().min().min()
print(f"⭐ run={nb_name}, N={c}")
print()
print(df_res[::-1].T[::-1].T.round(3).to_markdown()
      )
print()
print('args =', args)         
