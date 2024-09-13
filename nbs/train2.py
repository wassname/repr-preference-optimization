# %% [markdown]
# Instead of using the complex TRL we code it from scratch, using lighting
# 
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

# ML
from peft import LoraConfig, get_peft_model
from reprpo.models.load import load_model, print_trainable_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int, Bool
from torch.utils.data import DataLoader

from reprpo.gen import get_model_generations
from reprpo.helpers.shypothesis import shypothesis
from reprpo.evaluate import evaluate_adapters
from reprpo.data.collate3 import TokenizeRow

from open_pref_eval.evaluation import evaluate_model
from open_pref_eval.plot.radar import radar_plot
from open_pref_eval.datasets.genies import dist2datasets, GENIES
from open_pref_eval.datasets.ethics import get_ethics_datasets
from open_pref_eval.datasets import load_dataset_n
from open_pref_eval.datasets import ds2name
from open_pref_eval.plot.radar import radar_plot

from peft.tuners import BOFTConfig, OFTConfig, LoraConfig, IA3Config

from dataclasses import dataclass

import wandb
from datasets import load_dataset
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
from reprpo.train import Methods


from reprpo.train.dpo import compute_dpo_loss_batch, PL_DPO_MODEL

# %%



from reprpo.train.pl_base import TrainingArguments
from typing import Union

# get a union class from the enum
MethodsUnion = Union[tuple(e.value for e in Methods)]

def train(training_args:MethodsUnion):
    torch.set_float32_matmul_precision("high")

    PL_MODEL = training_args._reprpo_class
    model_kwargs = {k:getattr(training_args, k) for k in training_args._model_keys}

    ts = pd.Timestamp.now().strftime("%H%M%S")
    adapter_name = type(args).__name__
    group_name = f"{adapter_name}-{args.dataset}"
    run_fname = f'{adapter_name}/{ts}' # short for wandb
    wandb.require(experiment='service')

    config = dict(args=args, training_args=training_args)
    run = wandb.init(project=f'reprpo', name=run_fname, entity='wassname', group=f'{args.dataset}-{training_args.model_name.replace("/","")}', config=config)


    model, tokenizer = load_model(training_args.model_name, load_in_4bit=training_args.load_in_4bit,  load_in_8bit=training_args.load_in_8bit,  
                                attn_implementation='eager' # for gemma
    )

    # ### Load adapter
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
    model = get_peft_model(model, peft_config, adapter_name=group_name)
    print_trainable_parameters(model)
    if args.verbose:
        print(model)

    # ## Load data
    dataset2 = load_dataset("wassname/genie_dpo", name=args.dataset)


    # ### Data Loader
    # We use huggingface datasets, which are pretokenized. So that we can stack
    # from reprpo.data.collate import DPODataCollatorWithPadding, tokenize_row

    tokenize_row = TokenizeRow(tokenizer, max_length=training_args.max_length, max_prompt_length=training_args.max_prompt_length)

    if args.dev:
        # no cache
        import datasets
        datasets.disable_caching()
    dataset3 = dataset2.map(tokenize_row, batched=False)

    if args.verbose:
        print(f"Prompts truncated {np.mean(dataset3['train']['prompt_truncated']):2.2%}")
        print(f"Chosens truncated {np.mean(dataset3['train']['chosen_truncated']):2.2%}")


    # %%

    from transformers.data.data_collator import default_data_collator
    ds = dataset3
    dl_train = DataLoader(ds['train'].select_columns(['chosen', 'rejected', 'chosen_mask', 'rejected_mask']).with_format("torch"), batch_size=training_args.batch_size, 
                        #   collate_fn=default_data_collator
                        )

    dl_val = DataLoader(ds['test'].select_columns(['chosen', 'rejected', 'chosen_mask', 'rejected_mask']).with_format("torch"), batch_size=training_args.batch_size
                        # , collate_fn=default_data_collator
                        )

    if args.verbose:

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
    max_steps = training_args.n_samples // training_args.batch_size
    print('max optimiser steps', max_steps)

    # %%
    ideal_batch_size = max(16, training_args.batch_size) # probobly wont be stable with less than 16, so make up the difference with gradient accumulation
    accumulate_grad_batches = np.ceil(ideal_batch_size/training_args.batch_size).astype(int)
    print('accumulate_grad_batches', accumulate_grad_batches)
    print('accumulated batch size', training_args.batch_size*accumulate_grad_batches)

    print(f"epochs {training_args.n_samples//len(dl_train)}")

    # %%
    from lightning.pytorch.callbacks import LearningRateMonitor
    from reprpo.train.pl_base import GenCallback


    # %%
    pl_model = PL_MODEL(model,
                    # adam8bit=training_args.load_in_4bit or training_args.load_in_8bit, # saved mem, but seems unstable
                    schedule='constant',
                    weight_decay=training_args.weight_decay,
                    lr=training_args.lr,
                    num_iterations=max_steps,
                    batch_size=training_args.batch_size,

                    **model_kwargs
                    )




    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path(__file__).parent.parent
    model_fname= "_".join([training_args.model_name.replace("/", "_"), adapter_name, args.dataset])
    save_dir = root_dir / "outputs" / f"{model_fname}" / f"{timestamp}"
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    callbacks=[
                LearningRateMonitor(logging_interval='step'),
                # checkpoint_callback
            ]
    if args.verbose:
        callbacks+=[GenCallback(every=max_steps//2)]

    trainer = pl.Trainer(
            max_steps=max_steps,
            limit_val_batches=10,
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

            fast_dev_run=args.dev,
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

    if not args.dev:
        df_hist = read_metrics_csv(trainer.logger.experiment.metrics_file_path).bfill().ffill()
        # print(df_hist)

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


    # eval on ethics, GENIES, and our train dataset
    N = None
    if args.dev:
        N = 16
    datasets = [
        load_dataset_n('wassname/genie_dpo', name=args.dataset, split='train', N=N),
        load_dataset_n('wassname/genie_dpo', name=args.dataset, split='test', N=N),
    ]
    datasets += dist2datasets(GENIES, N=N, source=[args.dataset]) # our hard OOS test
    # datasets += get_ethics_datasets(N=N)
    datasets += [load_dataset_n('wassname/genie_dpo', name=name, split='test', N=N) for name in ['code_hard', #'truthful_qa',# 'wrong_arc', 'ranking_logic',
    # 'math', 'sycophancy_mimicry'
    ]]
    # print('datasets', [ds2name(d) for d in datasets])

    clear_mem()
    res, df_res2 = evaluate_model(model=model, 
                                tokenizer=tokenizer, 
                                datasets=datasets,
                                    batch_size=args.batch_size//4,
                                    bf16=True,
                                    torch_empty_cache_steps=33,)


    # %%

    df_res = df_res2.groupby(['dataset', 'adapter'], dropna=False)['correct'].mean().unstack().T
    df_res.columns = [d.replace('genie_dpo-', '') for d in df_res.columns]


    def key_metrics(df_res2):
        ds_name_train = ds2name(datasets[0])
        ds_name_test = ds2name(datasets[1])
        ds_name_oos = ds2name(datasets[2])
        ds_name_rnd = ds2name(datasets[3])

        df_res = df_res2.groupby(['dataset', 'adapter'], dropna=False)['correct'].mean().unstack().T
        rel_acc = df_res.loc[group_name]/df_res.loc['base']

        # metric: do we retrain train coherency?
        df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_chosen_logps'].mean().unstack().T
        rel_coherency = df_res_logp.loc[group_name]-df_res_logp.loc['base']

        # metric: do we retrain train coherency?
        c = df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_chosen_logps'].mean().unstack().T.loc[group_name]
        r = df_res_logp = df_res2.groupby(['dataset', 'adapter'], dropna=False)['_rejected_logps'].mean().unstack().T.loc[group_name]
        cho_rej_coh = c-r

        def fmt(s):
            return s.replace('genie_dpo-', '')
        
        # TODO make multiple cols of index

        
        df_metrics = pd.DataFrame([
            # accuracy increase over base measured generalisaton on increasing distribution shifts
            ['acc[pi/base]', 'train', fmt(ds_name_train), rel_acc[ds_name_train]],
            ['acc[pi/base]', 'test', fmt(ds_name_test), rel_acc[ds_name_test]],
            ['acc[pi/base]', 'oos', fmt(ds_name_oos), rel_acc[ds_name_oos]],
            ['acc[pi/base]', 'rnd', fmt(ds_name_rnd), rel_acc[ds_name_rnd]], # probobly wont go up as it's unrelated

            # we want to see if it retains coherency vs the base on chosen answers
            ['coherency[pi-base]', 'train', fmt(ds_name_train), rel_coherency[ds_name_train]],
            ['coherency[pi-base]', 'test', fmt(ds_name_test), rel_coherency[ds_name_test]],
            ['coherency[pi-base]', 'oos', fmt(ds_name_oos), rel_coherency[ds_name_oos]],
            ['coherency[pi-base]', 'rnd', fmt(ds_name_rnd), rel_coherency[ds_name_rnd]], 

            # we want to see if it retains chosen vs rejected
            ['coherency[cho-rej]', 'train', fmt(ds_name_train), cho_rej_coh[ds_name_train]],
            ['coherency[cho-rej]', 'test', fmt(ds_name_test), cho_rej_coh[ds_name_test]],
            ['coherency[cho-rej]', 'oos',  fmt(ds_name_oos), cho_rej_coh[ds_name_oos]],
            ['coherency[cho-rej]', 'rnd',  fmt(ds_name_rnd), cho_rej_coh[ds_name_rnd]],
        ], columns=['metric', 'split', 'dataset', 'value'])[['metric', 'split', 'value', 'dataset']]
        df_metrics = df_metrics.set_index(['metric', 'split'])
        df_metrics = df_metrics['value'].unstack()
        df_metrics.index.name = f'{adapter_name}\dist shift'
    
        return df_metrics

    # %%

    df_gen = get_model_generations(model, tokenizer, N=4)
    df_gen_w = wandb.Table(dataframe=df_gen.reset_index())


    df_metrics = key_metrics(df_res2)

    df_metrics_w = wandb.Table(dataframe=df_metrics.reset_index())


    df_res_w = wandb.Table(dataframe=df_res.reset_index())

    run.log({
        "acc": df_res_w,
        'relative_metrics': df_metrics_w,
        'generations': df_gen_w,
        
    })

    # save
    f = str(save_dir)+'/eval.parquet'
    df_res.to_parquet(f)
    # print(f'saved results to {f}')


    # %%
    from pprint import pprint
    from collections import OrderedDict
    ds_alias = OrderedDict(list(zip(['train', 'test', 'oos', 'rnd'], [ds2name(d) for d in datasets])))

    print(f'save_dir={save_dir}') 
    print('args =')
    pprint(args.__dict__)


    print(df_metrics.round(3).to_markdown())
    print("""Table 1: Key metrics (adapter over base model)\n""")

    cols = [v.replace('genie_dpo-','') for v in ds_alias.values()]
    df_res2 = df_res[cols]
    df_res2.columns = list(ds_alias.keys())
    df_res2.index.name = 'adapter/ds'
    print(df_res2.round(3).to_markdown())
    print("""Table 2: Absolute accuracy\n""")

    df_final = df_metrics.loc['acc[pi/base]'].to_frame(adapter_name).T
    df_final.index.name = 'acc_inc/eval_ds'
    print(df_final.round(3).to_markdown())
    print(f"""Table 3: Accuracy increase after training with adapter on `{args.dataset}` compared to base model `{training_args.model_name}` for various distribution shifts:""")
    for k,v in ds_alias.items():
        print(f"- `{k}`: `{v}`")



import tyro



import yaml, os
if __name__ == '__main__':

    # we can load a default config by passing it into the env
    # REPR_CONFIG=../configs/dev.yaml
    default_config = {}
    if os.environ.get('REPR_CONFIG') is not None:
        default_config = yaml.safe_load(os.environ.get('REPR_CONFIG'))   
        print(f'loaded default config from {os.environ.get("REPR_CONFIG")}')     
    
    MethodsUnion = Union[tuple(e.value for e in Methods)]
    args = tyro.cli(MethodsUnion, default=default_config)
    train(args)
