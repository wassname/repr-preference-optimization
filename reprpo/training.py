# Instead of using the complex TRL we code it from scratch, using lighting
#
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

import json
import os
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

# from matplotlib import pyplot as plt
# lightning
import lightning as pl

# Numeric
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

import warnings
# get a union class from the enum

from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from reprpo.helpers.pl_gen import GenCallback

from open_pref_eval.datasets import ds2name, load_dataset_n
from open_pref_eval.datasets.genies import GENIES, dist2datasets
from open_pref_eval.evaluation import evaluate_model

# ML
from peft import LoraConfig, get_peft_model
from peft.tuners import LoraConfig
from torch.utils.data import DataLoader

import wandb
from reprpo.data.collate3 import TokenizeRow
from reprpo.gen import display_gen, get_model_generations
from reprpo.helpers.lightning_hist import read_metrics_csv
from typing import Optional, Tuple, Union
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback
import yaml

# Local
from reprpo.helpers.torch import clear_mem
from reprpo.models.load import load_model, print_trainable_parameters
from .silence import silence, remove_warnings
from loguru import logger

remove_warnings()

# LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
LOGURU_FORMAT = "<level>{message}</level>"
logger.remove()
logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="INFO")


def apply_cfg_overrides(cfg, f=None):
    proj_root = Path(__file__).parent.parent
    overrides = {}
    if f is None:
        f = os.environ.get("REPR_CONFIG")
    if f is not None:
        f = proj_root / f
        print("applying REPR_CONFIG", f)
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"Warning: {k} not found in training_args")
        print(f"loaded default config from {f}")
    return cfg

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_display_name_from_args(args):
    """extract a human readable name from non-default args"""
    defaults = type(args)()
    # TODO need to init subclasses
    for k, v in asdict(defaults).items():
        if type(v).__name__ == "type":
            setattr(defaults, k, v())

    # collapse dict

    def list2tuple(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                list2tuple(v)
            elif isinstance(v, list):
                d[k] = tuple(v)
        return d

    diff = set(list2tuple(flatten_dict(asdict(args))).items())-set(list2tuple(flatten_dict(asdict(defaults))).items())
    diff = sorted(diff, key=lambda x: x[0])
    blacklist = ['eval_samples', 'base_model', 'dev', 'verbose', 'n_samples', 'batch_size', 'max_length', 'max_prompt_length', 'use_gradient_checkpointing', 'load_in_4bit', 'load_in_8bit', 'collection_keys_in', 'collection_keys_out', 'collection_hs', 'collection_layers_side', 'collection_layers_hs', 'save', 'wandb',]
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2g}"
        return v
    s = ' '.join([f"{k}={fmt(v)}" for k,v in list(diff) if k not in blacklist])

    cls_name = type(args).__name__
    if hasattr(args, 'transform'):
        cls_name += f"_{type(args.transform).__name__.replace('Transform','')}"
    if hasattr(args, 'loss'):
        cls_name += f"_{type(args.loss).__name__.replace('Loss','')}"
    cls_name = cls_name.replace('Config', '')

    # also either state transform and loss, or replace the words
    def rename(s):
        if hasattr(args, 'loss'):
            loss_name = type(args.loss).__name__.lower().replace('config', '').replace('loss', '')
            s = s.replace('loss.', f"{loss_name}.")
        if hasattr(args, 'transform'):
            transform_name = type(args.transform).__name__.lower().replace('config', '')
            s = s.replace('transform.', f"{transform_name}.")
        return s

    s_all = ' '.join([f"{k}={fmt(v)}" for k,v in list(diff)])
    s_short = f'{cls_name} {s}'
    s_all = rename(s_all)
    s_short = rename(s_short)
    logger.info(f"diff: {cls_name} {s_all}")

    return s_short


def safe_fn(s):
    """make a safe filename from any string."""
    return "".join(c for c in s if c.isalpha() or c.isdigit() or c==' ' or c=="_" or c=="-").rstrip()


def train(args, trial: Optional[Trial] = None):
    if args.verbose < 1:
        silence()
    torch.set_float32_matmul_precision("medium")

    PL_MODEL = args._cls

    logger.info(f"PL_MODEL {PL_MODEL}")

    ds_name_train = args.dataset.replace("genies_preferences-", "")
    model_name = args.base_model.split("/")[-1]

    ts = pd.Timestamp.now().strftime("%H%M%S")
    adapter_name = args._name
    # adapter_name = get_args_dict(args)

    human_name = get_display_name_from_args(args) # f"{adapter_name}_{ds_name_train}"

    # we can set an experiment group name from env vars, otherwise it will just groupt by model and training ds
    group_name = f"{ds_name_train}-{model_name}"
    if os.environ.get("WANDB_GROUP", None) is not None:
        group_name = safe_fn(os.environ.get("WANDB_GROUP") + "-" + group_name)
        logger.info(f"Using WANDB_GROUP=https://wandb.ai/wassname/reprpo2/groups/{group_name} ")
    if args.verbose > 1:
        logger.info("args")
        pprint(args, compact=True)
        # logger.info("model_kwargs", model_kwargs.keys())

        logger.info(f"Using finetune_name={human_name}")

    run_fname = f"{adapter_name}/{ts}"  # short for wandb

    # save_dir
    timestamp = safe_fn(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    root_dir = Path(__file__).parent.parent
    model_fname = safe_fn("_".join(
        [args.base_model.replace("/", "-"), adapter_name, ds_name_train]
    ))
    save_dir = root_dir / "outputs" / group_name / f"{model_fname}" / f"{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    config = asdict(args)
    config.update({'post': {
        'group_name': group_name,
        'adapter_name': adapter_name,
        'human_name': human_name,
        'model_fname': model_fname,
        'ds_name_train': ds_name_train,
        'run_fname': run_fname,
        'save_dir': str(save_dir),
    'ts': ts,}})
    if args.wandb:
        with warnings.catch_warnings(action='ignore'):

            wandb.require(experiment="core")
            pl_wandb_logger=WandbLogger(
                name=run_fname, save_dir=save_dir,
                project="reprpo2",
                # entity="wassname",
                group=group_name,
                # config=config,
                mode="disabled"
                if os.environ.get("WANDB_MODE", None) == "disabled"
                else "online",
            )

        # in case we already initialised it earlier, update it
        wandb.config.update(config)
        wandb.run.tags = tuple(wandb.run.tags) + (
            ds_name_train, 
            model_fname, 
            adapter_name
        )
        wandb.run.name = run_fname
        # wandb.run.groupName = group_name
        wandb.run._quiet = True
    # run = pl_wandb_logger._experiment

    # config
    (save_dir / "config.json").open("w").write(json.dumps(config, indent=4))

    # logging
    if args.verbose < 1:
        logger.remove() # info by default
        logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="WARNING")
    log_file = save_dir / "log.txt"
    logger.add(
        log_file,
        level="INFO",
        format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}",
    )

    model, tokenizer = load_model(
        args.base_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        # attn_implementation='eager' # for gemma
    )

    # ### Load adapter
    """
    Note that GENIES and PEFT use the default targets for llama ["q_proj", "v_proj"]
    but other papers like qlora and HRA use all linear layers
    """
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        use_rslora=True,
        # use_dora=True,
        task_type="CAUSAL_LM",
        # target_modules=["all-linear"], #  QLoRA-style training
    )
    # if hasattr(PL_MODEL, 'setup_grad_proj'):
    #     peft_config = PL_MODEL.setup_grad_proj(peft_config)
    
    model = get_peft_model(model, peft_config, adapter_name=adapter_name)
    print_trainable_parameters(model)
    if args.verbose > 1:
        logger.info(f"{model}")

    if args.dev:
        # no cache
        import datasets
        datasets.disable_caching()
    tokenize_row = TokenizeRow(
        tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # ## Load data
    ds_train = load_dataset("wassname/genies_preferences", name=args.dataset)
    ds_train_tok = ds_train.map(tokenize_row, batched=False)    

    if args.verbose > 0:
        logger.info(
            f"Prompts truncated {np.mean(ds_train_tok['train']['prompt_truncated']):2.2%}"
        )
        logger.info(
            f"Chosens truncated {np.mean(ds_train_tok['train']['chosen_truncated']):2.2%}"
        )

    def ds2dl(ds):
        return DataLoader(
            ds
            .select_columns(["chosen", "rejected", "chosen_mask", "rejected_mask"])
            .with_format("torch"),
            batch_size=args.batch_size,
        )

    dl_train = ds2dl(ds_train_tok["train"])

    # do we want to use test or OOS as val? I choose OOS as I want to find the intervention that generalises the best (and then I validate on other distribution shifts to avoid cherry-picking/overfitting)
    ds_val_oos = dist2datasets(
        GENIES,
        N=150,
        source=[args.dataset],
    )[0].map(tokenize_row, batched=False)
    dl_val = ds2dl(
        ds_val_oos
    )

    if args.verbose > 2:
        logger.info("QC one train batch (after pad/crop")
        batch = next(iter(dl_train))
        # logger.info(batch.keys())
        # logger.info(tokenizer.decode(batch['prompt'][0]))
        logger.info("===")
        logger.info(f"{tokenizer.decode(batch['chosen'][0])}")
        logger.info("---")
        logger.info(f"{tokenizer.decode(batch['rejected'][0])}")
        logger.info("===")

    if wandb.run is not None:
        logger.info(f"WANDB url = {wandb.run.get_url()}")

    # ## Trainer
    # - https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
    # - https://gist.github.com/wassname/e29d02b5026a531e13912cf768e6fdc8

    max_steps = args.n_samples // args.batch_size

    ideal_batch_size = max(
        16, args.batch_size
    )  # probobly wont be stable with less than 16, so make up the difference with gradient accumulation
    accumulate_grad_batches = np.ceil(
        ideal_batch_size / args.batch_size
    ).astype(int)
    if args.verbose>1:
        logger.info(
            f"max optimiser steps {max_steps}",
        )
        logger.info(
            f"accumulate_grad_batches {accumulate_grad_batches}",
        )
        logger.info(
            f"accumulated batch size {args.batch_size * accumulate_grad_batches}"
        )
        logger.info(f"epochs {args.n_samples//len(dl_train.dataset)}")

    model_kwargs = {k: getattr(args, k) for k in args._model_keys}
    pl_model = PL_MODEL(
        model,
        adam8bit=args.load_in_4bit
        or args.load_in_8bit,  # saved mem, but seems unstable?
        schedule="wsd",
        num_iterations=max_steps,
        batch_size=args.batch_size,
        # model args
        **model_kwargs,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # checkpoint_callback
    ]
    if trial is not None:
        callbacks += [PyTorchLightningPruningCallback(trial, 'val/dpo_acc')]
    if args.verbose>1:
        callbacks += [GenCallback(every=max_steps // 2 + 1)]


    model_kwargs = {k: getattr(args, k) for k in args._model_keys}
    loggers = [CSVLogger(name=run_fname, save_dir=save_dir, flush_logs_every_n_steps=5)]
    if args.wandb:
        loggers.append(pl_wandb_logger)
    trainer = pl.Trainer(
        max_steps=max_steps,
        # limit_val_batches=max(6, max_steps//10),
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
        logger=loggers,
        default_root_dir=save_dir,
        # too large, we will just save adapter afterwards
        enable_checkpointing=False,
        fast_dev_run=args.dev,
        enable_progress_bar=args.verbose > 0,
        enable_model_summary=args.verbose > 1,
    )
    trainer.logger.log_hyperparams(config)

    # train
    trainer.fit(pl_model, dl_train, dl_val)

    # save as regular adapter only (small)
    if args.save:
        model.save_pretrained(
            str(save_dir / "adapter"),
        )
        logger.info(f"saved to {save_dir/'adapter'}")

    # ### Hist
    if not args.dev:
        df_hist = (
            read_metrics_csv(trainer.logger.experiment.metrics_file_path)
            .bfill()
            .ffill()
        )
        # logger.info(df_hist)

    # eval(model, tokenizer, args, save_dir, run, finetune_name, adapter_name)

    # ## Gen
    model.cuda()  # for some reason it ends up cpu

    if (not args.dev) and (args.verbose > 0):
        N = args.verbose
        df_gen = get_model_generations(model, tokenizer, N=N)
        display_gen(df_gen.head(N))

    logger.debug("eval")

    ds_val_oos = ds_train_tok = dl_train = dl_val = None
    trainer = None
    # ## Eval
    # eval on ethics, GENIES, and our train dataset
    N = args.eval_samples
    datasets = [
        load_dataset_n(
            "wassname/genies_preferences",
            name=args.dataset,
            split="train",
            N=750 if N is None else min(N, 750),
        ),
        load_dataset_n(
            "wassname/genies_preferences", name=args.dataset, split="test", N=N
        ),
    ]
    # datasets += [
    #     load_dataset_n("wassname/genies_preferences", name=name, split="test", N=N)
    #     for name in [
    #         # "math_make_questions",  #'truthful_qa',# 'wrong_arc', 
    #         'ranking_logic',  # FIXME make sure we choose a dataset that is random and not in 
    #         # 'math', 'sycophancy_mimicry'
    #     ]
    # ]

    # get the out of distribution set
    datasets += dist2datasets(
        GENIES,
        # N=N, # can't cheap out on the main metric
        source=[args.dataset],
    )  # our hard OOS test

    # our unrelated dataset
    datasets += [
        load_dataset_n('wassname/ethics_expression_preferences', name='justice', split='test', N=N)
    ]
    ds_names =  [ds2name(d) for d in datasets]
    logger.info(f"evaluating on {ds_names}")

    clear_mem()
    res, df_res2 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_size=args.batch_size,
        bf16=True,
        torch_empty_cache_steps=100,
        verbose=args.verbose,
        # dataloader_num_workers=2,
        # dataloader_pin_memory=True,
    )

    ds_alias = OrderedDict(
        list(zip(["train", "test", "oos", "rnd"], ds_names))
    )
    ds_alias_rev = {v: k for k, v in ds_alias.items()}
    df_res2['ds_alias'] = df_res2['dataset'].map(lambda x: ds_alias_rev.get(x, x))

    # save
    f = str(save_dir) + "/eval.parquet"
    df_res2.to_parquet(f)
    logger.info(f"save_dir={save_dir}")
    # pprint(args, compact=1)

    r = parse_eval(df_res2, ds_alias, human_name=human_name, base_model=model_name)

    # WANDB logging
    r2 = {}
    for k, v in r.items():
        r2[k] = wandb.Table(dataframe=v.reset_index())
        # log first row too, so we can compare single value
        if wandb.run is not None:
            wandb.log({f"res/{k}/{kk}": vv for kk, vv in v.iloc[0, :].to_dict().items()})
        # also just log final metrics to wandb so we can view a group

    if wandb.run is not None:
        if (not args.dev) and (args.verbose > 0):
            df_gen_w = wandb.Table(dataframe=df_gen)
            wandb.log({"generations": df_gen_w, **r2})

        logger.info(f"WANDB url = {wandb.run.get_url()}")

    # return a single dict for hyperparam tuning
    dd = [
        {f"{metric}/{split}": v for split, v in df.iloc[0].to_dict().items()}
        for metric, df in r.items()
    ]
    rd = {}
    [rd.update(**ddd) for ddd in dd]

    wandb.finish(quiet=True)
    return rd


def key_metrics(df_res2, adapter_name, ds_alias):
    # adapter_name, finetune_name, ds_alias
    ds_name_train = ds_alias["train"]
    ds_name_test = ds_alias["test"]
    ds_name_oos = ds_alias["oos"]
    ds_name_rnd = ds_alias["rnd"]

    # main metric, accuracy, how often the prefered answer
    df_res = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["correct"]
        .mean()
        .unstack()
        .T
    )
    acc_gain_vs_ref = df_res.loc[adapter_name] / df_res.loc["base"]

    # metric: do we retain coherency? measured with perplexity
    d = df_res2.set_index(['dataset', 'ds_i'])[['adapter', '_chosen_ppl']]
    perplexity_reduction_vs_ref = (np.log(d.query('adapter == "base"')['_chosen_ppl']) - np.log(d.query('adapter == @adapter_name')['_chosen_ppl']))
    perplexity_reduction_vs_ref = np.exp(perplexity_reduction_vs_ref.reset_index().groupby('dataset')['_chosen_ppl'].mean())

    # metric: how much does the model follow the preference vs the rejected. log ratio difference or ρθ in GPO https://arxiv.org/html/2402.05749v2
    df_res2["logratios"] = df_res2["_chosen_logps"] - df_res2["_rejected_logps"]
    df_logratios = (
        df_res2.groupby(["dataset", "adapter", "ds_i"])["logratios"].mean().unstack(0)
    )
    model_logratios = df_logratios.loc[adapter_name]
    ref_logratios = df_logratios.loc["base"]
    preference_logp_gain_vs_ref = (model_logratios - ref_logratios).mean()

    def fmt(s):
        return s.replace("genies_preferences-", "")

    def generate_metrics(metric_name, datasets, values):
        o = []
        for split, ds_name in datasets.items():
            if ds_name in values:
                o.append([metric_name, split, fmt(ds_name_train), values[ds_name]])
        return o

    # Define the datasets
    datasets = OrderedDict(
        train=ds_name_train,
        test=ds_name_test,
        oos=ds_name_oos,
        rnd=ds_name_rnd,
    )

    # Generate metrics for each category
    acc_gain_vs_ref_metrics = generate_metrics(
        "acc_gain_vs_ref", datasets, acc_gain_vs_ref
    )
    perplexity_reduction_vs_ref_metrics = generate_metrics(
        "perplexity_reduction_vs_ref", datasets, perplexity_reduction_vs_ref
    )
    preference_logp_gain_vs_ref_metrics = generate_metrics(
        "preference_logp_gain_vs_ref", datasets, preference_logp_gain_vs_ref
    )

    # Concatenate all metrics
    all_metrics = (
        acc_gain_vs_ref_metrics
        + perplexity_reduction_vs_ref_metrics
        + preference_logp_gain_vs_ref_metrics
    )

    # Create DataFrame
    df_metrics = pd.DataFrame(
        all_metrics, columns=["metric", "split", "dataset", "value"]
    )[["metric", "split", "value", "dataset"]]
    df_metrics = df_metrics.set_index(["metric", "split"])
    df_metrics = df_metrics["value"].unstack()
    df_metrics.index.name = f"{adapter_name}\dist shift"

    return df_metrics.iloc[:, ::-1]


def parse_eval(df_res2, ds_alias, human_name, base_model="", verbose=True):
    adapter_name = df_res2[["adapter"]].query('adapter!="base"').values[0, 0]
    ds_alias_rev = {v:k for k,v in ds_alias.items()}

    df_res = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["correct"]
        .mean()
        .unstack()
        .T
    )
    df_res = df_res.rename(columns=ds_alias_rev)

    df_metrics = key_metrics(df_res2, adapter_name, ds_alias)
    if verbose:
        logger.info(f"\n{df_metrics.round(3).to_markdown()}")
        logger.info("""Table 1: Key metrics (adapter over base model)\n""")

    cols = [v.replace("genies_preferences-", "") for v in ds_alias.values()]
    df_res2 = df_res[list(ds_alias.keys())]
    df_res2.index.name = "adapter/ds"
    if verbose:
        logger.info(f"\n{df_res2.round(3).to_markdown()}")
        logger.info("""Table 2: Absolute accuracy\n""")

    df_final = df_metrics.loc["acc_gain_vs_ref"].to_frame(human_name).T
    df_final = df_final * 100 - 100  # percentage points
    df_final.index.name = "acc_inc/eval_ds [pp]"
    caption = f"""Table 3🥇: Accuracy increase (in percentage points) after training with named adapter on ds:`{ds_alias["train"]}` compared to base model `{base_model}` for various distribution shifts:"""
    for k, v in ds_alias.items():
        caption += f"\n- `{k}`: `{v}`"
    if verbose:
        print(f"\n{df_final.round(3).to_markdown()}")
        logger.info(caption)


    if not df_metrics['train']['acc_gain_vs_ref']>=1.0:
        logger.error(f"Worse `acc` on training set for `{human_name}` (didn't learn?)")

    # now one comparing acc train vs test in df_res2
    if not df_res2['train'][adapter_name] >= df_res2['test'][adapter_name]:
        logger.error(f"Underfitting (acc_test>ac_train) for `{human_name}` (train for longer?)")

    # https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
    # this one often happens as dpo makes preference better and ppx worse
    if not df_metrics['train']['perplexity_reduction_vs_ref']>=105:
        # the ppx always gets worse with dpo, but usually only like 10
        # FIXME I indented this to indicate incoherence but it doesn't seem to track
        logger.warning(f"Worse `ppx` on training set for `{human_name}` (incoherent model? loss to high?)")
    if not df_metrics['train']['preference_logp_gain_vs_ref']>=0:
        logger.error(f"Worse `pref` on training set for `{human_name}` (didn't learn?)")

    df_acc = df_res2.loc[adapter_name].to_frame(human_name).T
    info = {
        "acc": df_acc,
        "acc_gain_vs_ref": df_final,
    }

    # format for wandb. just one row, one data type per table
    for i in df_metrics.index:
        info[i] = df_metrics.loc[i].to_frame(human_name).T
        info[i].index.name = 'model' # # the index becomes a new col

    return info
