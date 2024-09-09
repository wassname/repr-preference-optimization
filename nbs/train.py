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

import wandb

# Numeric
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt

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


from reprpo.train.dpo import compute_dpo_loss_batch, PL_DPO_MODEL

# %%
torch.set_float32_matmul_precision("high")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# %%
from simple_parsing import ArgumentParser
from dataclasses import dataclass

@dataclass
class CLIArguments:
    method: str = 'dpo' # reprpo_svd # reprpo_side
    # dataset: str = 'code_easy'
    dataset: str = 'us_history_textbook'
    verbose: bool = False
    dev: bool = False


parser = ArgumentParser()
parser.add_arguments(CLIArguments, dest='cli')
# parser.add_argument('-m', '--method', type=str, default='dpo', help='dpo, reprpo_svd, reprpo_side')
# parser.add_argument('-d', '--dataset', type=str, default='code_easy', help='code_easy etc see subsets in https://huggingface.co/datasets/wassname/genie_dpo')
# parser.add_argument('-v', '--verbose', type=bool, default=False, action="store_true", help='print dataset')
# parser.add_argument('--dev', type=bool, default=False, action="store_true", help='fast dev run')
args1 = parser.parse_known_args()[0].cli


if args1.method == 'dpo':
    from reprpo.train.dpo import DPOTrainingArguments as TrainingArguments, PL_DPO_MODEL as PL_MODEL
elif args1.method == 'reprpo_svd':
    from reprpo.train.reprpo_svd import ReprPOSVDTrainingArguments as TrainingArguments, PL_REPRPO_SVD_MODEL as PL_MODEL
elif args1.method == 'reprpo_side':
    from reprpo.train.reprpo_side import ReprPOSideInTrainingArguments as TrainingArguments, PL_REPRPO_SIDE_MODEL as PL_MODEL
    # from reprpo.train.reprpo_side import ReprPOSideOutTrainingArguments as TrainingArguments, PL_REPRPO_SIDE_MODEL as PL_MODEL
elif args1.method == 'reprpo_ortho':
    from reprpo.train.reprpo_ortho import ReprPOOrthoTrainingArguments as TrainingArguments, PL_REPRPO_ORTHO_MODEL as PL_MODEL
else:
    raise ValueError(f"method {args1.method} not found. options: `reprpo_side`, `dpo`, `reprpo_svd`")


parser.add_arguments(TrainingArguments, dest='args')
# parser.add_arguments(CLIArguments, dest='cli')
args2 = parser.parse_args()
args = TrainingArguments(**args2.args.__dict__)
print(PL_MODEL, TrainingArguments)
print(f"args = {args}")

if args1.dev:
    args.model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    # or  'yujiepan/llama-3-tiny-random'
    args.n_samples = 512
    args.batch_size *= 2

# %% [markdown]
# ## Load model

ts = pd.Timestamp.now().strftime("%H%M%S")
run_fname = f'{args.adapter_name}/{ts}'
wandb.require(experiment='service')


config = dict(**args1.__dict__, **args.__dict__)
run = wandb.init(project=f'reprpo', name=run_fname, entity='wassname', group=f'{args1.dataset}-{args.model_name.replace("/","")}', config=config)

from peft import LoraConfig, get_peft_model
from reprpo.models.load import load_model, print_trainable_parameters


model, tokenizer = load_model(args.model_name, load_in_4bit=args.load_in_4bit,  load_in_8bit=args.load_in_8bit,  
                              attn_implementation='eager' # for gemma
)

# %% [markdown]
# ### Load adapter

# %%
from peft.tuners import BOFTConfig, OFTConfig, LoraConfig, IA3Config


adapter_name = f"{args.adapter_name}-{args1.dataset}"

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    # lora_dropout=0.1,  # Changed
    # bias="none",
    # lora_alpha=16, 
    # r=16,
    use_rslora=True,
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
if args1.dev:
    dataset2['train'] = dataset2['train'].select(range(16))
    dataset2['test'] = dataset2['test'].select(range(16))



# %% [markdown]
# ### Data Loader
# 
# We use huggingface datasets, which are pretokenized. So that we can stack

# %%


# %%
# from reprpo.data.collate import DPODataCollatorWithPadding, tokenize_row
from reprpo.data.collate3 import TokenizeRow
tokenize_row = TokenizeRow(tokenizer, max_length=args.max_length, max_prompt_length=args.max_prompt_length)

if args1.dev:
    # no cache
    import datasets
    datasets.disable_caching()
dataset3 = dataset2.map(tokenize_row, batched=False)

if args1.verbose:
    print(f"Prompts truncated {np.mean(dataset3['train']['prompt_truncated']):2.2%}")
    print(f"Chosens truncated {np.mean(dataset3['train']['chosen_truncated']):2.2%}")


# %%

from transformers.data.data_collator import default_data_collator
ds = dataset3
dl_train = DataLoader(ds['train'].select_columns(['chosen', 'rejected', 'chosen_mask', 'rejected_mask']).with_format("torch"), batch_size=args.batch_size, 
                    #   collate_fn=default_data_collator
                      )

dl_val = DataLoader(ds['test'].select_columns(['chosen', 'rejected', 'chosen_mask', 'rejected_mask']).with_format("torch"), batch_size=args.batch_size
                    # , collate_fn=default_data_collator
                    )

if args1.verbose:

    print('QC one dataset row')
    r = dataset2['train'][0]
    print(r['prompt'])
    print('===')
    print(r['chosen'])
    print('---')
    print(r['rejected'])
    print()
    print()

    print('QC one train batch (after pad/crop')
    batch = next(iter(dl_train))
    # print(batch.keys())
    # print(tokenizer.decode(batch['prompt'][0]))
    print('===')
    print(tokenizer.decode(batch['chosen'][0]))
    print('---')
    print(tokenizer.decode(batch['rejected'][0]))
    print()
    print()


    


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

if args1.method == "reprpo_svd":
    model_kwargs = dict(
        alpha=args.alpha,
        quantile=args.quantile,
        dual_svd=args.dual_svd,
        collection_layers=args.collection_layers,
    )
elif args1.method == "reprpo_side":
    model_kwargs = dict(
        alpha=args.alpha,
        collection_layers=args.collection_layers,
        collection_keys=args.collection_keys,
        collect_input=args.collect_input,
    )
elif args1.method == "reprpo_ortho":
    model_kwargs = dict(
        alpha=args.alpha,
        collection_layers=args.collection_layers,
    )
else:
    model_kwargs = dict()


# %%
pl_model = PL_MODEL(model,
                # adam8bit=args.load_in_4bit or args.load_in_8bit,
                schedule='constant',
                weight_decay=args.weight_decay,
                lr=args.lr,
                num_iterations=max_steps,
                batch_size=args.batch_size,

                **model_kwargs
                )


# %%
# from reprpo.helpers.lightning_save import AdapterModelCheckpoint

# checkpoint_callback = AdapterModelCheckpoint(
#     verbose=True,
# )

# %%



timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
root_dir = Path(__file__).parent.parent
model_fname= "_".join([args.model_name.replace("/", "_"), args.adapter_name, args1.dataset])
save_dir = root_dir / "outputs" / f"{model_fname}" / f"{timestamp}"
Path(save_dir).mkdir(exist_ok=True, parents=True)

callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # checkpoint_callback
        ]
if args1.verbose:
    callbacks+=[GenCallback(every=max_steps//2)]

trainer = pl.Trainer(
        max_steps=max_steps,
        limit_val_batches=10,
#         limit_val_batches=max_batches//5,
        gradient_clip_val=0.3,

        # accelerator='gpu',
        devices=1, 
        # plugins=plugins,
        
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html
        # mixed wont convert the modules weights, while bf16-true would
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "f16-mixed",
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=[
            CSVLogger(name=run_fname, save_dir=save_dir, flush_logs_every_n_steps=5),
            WandbLogger(name=run_fname, save_dir=save_dir),
        ],
        default_root_dir=save_dir,

        # too large, we will just save adapter afterwards
        enable_checkpointing=False, 

        fast_dev_run=args1.dev,
    )

# train
trainer.fit(pl_model, dl_train, dl_val)

# %%
# save as regular adapter only

model.save_pretrained(
    str(save_dir)+'-adapter',
)
print(f'saved to {save_dir}-adapter')


# %% [markdown]
# ### Hist

# %%

if not args1.dev:
    df_hist = read_metrics_csv(trainer.logger.experiment.metrics_file_path).bfill().ffill()
    print(df_hist)

# import matplotlib
# plt.style.use('ggplot')
# matplotlib.rcParams['figure.figsize'] = (6, 2)
# plot_hist(df_hist, ['.*/loss_step', '.*/acc.*', '.*/auroc.*', '.*/.*reward_step'])
# todo val and train seperate for few epochs


# %% [markdown]
# ## Gen

# %%
model.cuda(); # for some reason it ends up cpu



# %% [markdown]
# ## Eval

# %%
from reprpo.helpers.shypothesis import shypothesis
from reprpo.evaluate import evaluate_adapters
from open_pref_eval.evaluation import evaluate_model
from open_pref_eval.plot.radar import radar_plot
from open_pref_eval.datasets.genies import dist2datasets, GENIES
from open_pref_eval.datasets.ethics import get_ethics_datasets
from open_pref_eval.datasets import load_dataset_n

# eval on ethics, GENIES, and our train dataset
N = None
if args1.dev:
    N = 16
datasets = [
    load_dataset_n('wassname/genie_dpo', name=args1.dataset, split='train', N=N),
    load_dataset_n('wassname/genie_dpo', name=args1.dataset, split='test', N=N),
]
datasets += dist2datasets(GENIES, N=N, source=[args1.dataset]) # our hard OOS test
# datasets += get_ethics_datasets(N=N)
datasets += [load_dataset_n('wassname/genie_dpo', name=name, split='test', N=N) for name in ['code_hard', #'truthful_qa',# 'wrong_arc', 'ranking_logic',
# 'math', 'sycophancy_mimicry'
]]
# print('datasets', [ds2name(d) for d in datasets])

clear_mem()
res, df_res2 = evaluate_model(model=model, 
                              tokenizer=tokenizer, 
                              datasets=datasets,
                                 batch_size=args2.args.batch_size//4,
                                 bf16=True,
                                 torch_empty_cache_steps=33,)


# %%
from open_pref_eval.datasets import ds2name
from open_pref_eval.plot.radar import radar_plot
df_res = df_res2.groupby(['dataset', 'adapter'], dropna=False)['correct'].mean().unstack().T


def key_metrics(df_res2):
    ds_name_train = ds2name(datasets[0])
    ds_name_test = ds2name(datasets[1])
    ds_name_oos = ds2name(datasets[2])
    ds_name_rnd = ds2name(datasets[3])

    df_res = df_res2.groupby(['dataset', 'adapter'], dropna=False)['correct'].mean().unstack().T
    rel_acc = df_res.loc[adapter_name]/df_res.loc['base']

    # metric: do we retrain train coherency?
    df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_chosen_logps'].mean().unstack().T
    rel_coherency = df_res_logp.loc[adapter_name]-df_res_logp.loc['base']

    # metric: do we retrain train coherency?
    c = df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_chosen_logps'].mean().unstack().T.loc[adapter_name]
    r = df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_rejected_logps'].mean().unstack().T.loc[adapter_name]
    cho_rej_coh = c-r

    def fmt(s):
        return s.replace('genie_dpo-', '')

    df_metrics = pd.Series({
        # accuracy increase over base measured generalisaton on increasing distribution shifts
        f'acc[a/base]_train [{fmt(ds_name_train)}]': rel_acc[ds_name_train],
        f'acc[a/base]_test [{fmt(ds_name_test)}]': rel_acc[ds_name_test],
        f'acc[a/base]_oos [{fmt(ds_name_oos)}]': rel_acc[ds_name_oos],
        f'acc[a/base]_rnd [{fmt(ds_name_rnd)}]': rel_acc[ds_name_rnd], # probobly wont go up as it's unrelated

        # we want to see if it retains coherency vs the base on chosen answers
        f'coherency[a-base]_train [{fmt(ds_name_train)}]': rel_coherency[ds_name_train],
        f'coherency[a-base]_test [{fmt(ds_name_test)}]': rel_coherency[ds_name_test],
        f'coherency[a-base]_oos [{fmt(ds_name_oos)}]': rel_coherency[ds_name_oos],
        f'coherency[a-base]_rnd [{fmt(ds_name_rnd)}]': rel_coherency[ds_name_rnd], 

        # we want to see if it retains chosen vs rejected
        f'coherency[cho-rej]_train [{fmt(ds_name_train)}]': cho_rej_coh[ds_name_train],
        f'coherency[cho-rej]_test [{fmt(ds_name_test)}]': cho_rej_coh[ds_name_test],
        f'coherency[cho-rej]_oos [{fmt(ds_name_oos)}]': cho_rej_coh[ds_name_oos],
        f'coherency[cho-rej]_rnd [{fmt(ds_name_rnd)}]': cho_rej_coh[ds_name_rnd], 

        
    })
    return df_metrics.to_frame('val')

# %%
from reprpo.gen import get_model_generations
df_gen = get_model_generations(model, tokenizer, N=4)
df_gen_w = wandb.Table(dataframe=df_gen.reset_index())


df_metrics = key_metrics(df_res2)
print('key metrics (adapter over base model)\n', df_metrics)
df_metrics_w = wandb.Table(dataframe=df_metrics.reset_index())

print('acc res')
print(df_res)
df_res_w = wandb.Table(dataframe=df_res.reset_index())

run.log({
    "acc": df_res_w,
    'relative_metrics': df_metrics_w,
    'generations': df_gen_w,
    
})

# save
f = str(save_dir)+'/eval.parquet'
df_res.to_parquet(f)
print(f'saved results to {f}')

# radar_plot(df_res)
df_res


# %%

# print acc for journal
c  = df_res2.groupby(['adapter', 'dataset']).count().min().min()
df_res.columns = [s.replace('genie_dpo-','') for s in df_res.columns]
print(f"⭐ run={run_fname}, N={c}")
print()
print(df_res.round(3).to_markdown()
      )
print()
print('args =', args)     
print(f'save_dir={save_dir}') 
