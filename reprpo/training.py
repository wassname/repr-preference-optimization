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

# Local
from reprpo.helpers.torch import clear_mem
from reprpo.models.load import load_model, print_trainable_parameters
from .silence import silence, remove_warnings
from loguru import logger

remove_warnings()

logger.remove()
# LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
LOGURU_FORMAT = "|<level>{level}</level>| {message}"
logger.add(os.sys.stdout, format=LOGURU_FORMAT, level="INFO")


def train(training_args):
    if training_args.verbose < 1:
        silence()
    torch.set_float32_matmul_precision("medium")

    PL_MODEL = training_args._cls

    logger.info(f"PL_MODEL {PL_MODEL}")

    ds_name_train = training_args.dataset.replace("genies_preferences-", "")
    model_name = training_args.base_model.split("/")[-1]

    ts = pd.Timestamp.now().strftime("%H%M%S")
    adapter_name = training_args._name

    finetune_name = f"{adapter_name}_{ds_name_train}"

    # we can set an experiment group name from env vars, otherwise it will just groupt by model and training ds
    group_name = f"{ds_name_train}-{model_name}"
    if os.environ.get("WANDB_GROUP", None) is not None:
        group_name = os.environ.get("WANDB_GROUP") + "_" + group_name
    if training_args.verbose > 0:
        logger.info("training_args")
        pprint(training_args, compact=True)
        # logger.info("model_kwargs", model_kwargs.keys())

        logger.info(f"Using WANDB_GROUP=https://wandb.ai/wassname/reprpo2/{group_name} ")
        logger.info(f"Using finetune_name={finetune_name}")

    run_fname = f"{adapter_name}/{ts}"  # short for wandb
    wandb.require(experiment="core")

    config = asdict(training_args)
    run = wandb.init(
        project="reprpo2",
        name=run_fname,
        entity="wassname",
        group=group_name,
        config=config,
        mode="disabled"
        if os.environ.get("WANDB_MODE", None) == "disabled"
        else "online",
    )

    # save_dir
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path(__file__).parent.parent
    model_fname = "_".join(
        [training_args.base_model.replace("/", "_"), adapter_name, ds_name_train]
    )
    save_dir = root_dir / "outputs" / f"{model_fname}" / f"{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    # config
    (save_dir / "config.json").open("w").write(json.dumps(config, indent=4))

    # logging
    log_file = save_dir / "log.txt"
    logger.add(
        log_file,
        level="INFO",
        format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}",
    )

    model, tokenizer = load_model(
        training_args.base_model,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
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
    
    model = get_peft_model(model, peft_config, adapter_name=finetune_name)
    print_trainable_parameters(model)
    if training_args.verbose > 1:
        logger.info(f"{model}")

    # ## Load data
    dataset2 = load_dataset("wassname/genies_preferences", name=training_args.dataset)

    # ### Data Loader
    # We use huggingface datasets, which are pretokenized. So that we can stack

    tokenize_row = TokenizeRow(
        tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
    )

    if training_args.dev:
        # no cache
        import datasets

        datasets.disable_caching()
    dataset3 = dataset2.map(tokenize_row, batched=False)

    if training_args.verbose > 0:
        logger.info(
            f"Prompts truncated {np.mean(dataset3['train']['prompt_truncated']):2.2%}"
        )
        logger.info(
            f"Chosens truncated {np.mean(dataset3['train']['chosen_truncated']):2.2%}"
        )

    ds = dataset3
    dl_train = DataLoader(
        ds["train"]
        .select_columns(["chosen", "rejected", "chosen_mask", "rejected_mask"])
        .with_format("torch"),
        batch_size=training_args.batch_size,
        #   collate_fn=default_data_collator
    )

    dl_val = DataLoader(
        ds["test"]
        .select_columns(["chosen", "rejected", "chosen_mask", "rejected_mask"])
        .with_format("torch"),
        batch_size=training_args.batch_size,
        # , collate_fn=default_data_collator
    )

    if training_args.verbose > 1:
        # logger.info("QC one dataset row")
        # r = dataset2["train"][0]
        # logger.info(r["prompt"])
        # logger.info("===")
        # logger.info(r["chosen"])
        # logger.info("---")
        # logger.info(r["rejected"])
        logger.info("===")
        # logger.info()

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

    max_steps = training_args.n_samples // training_args.batch_size

    ideal_batch_size = max(
        16, training_args.batch_size
    )  # probobly wont be stable with less than 16, so make up the difference with gradient accumulation
    accumulate_grad_batches = np.ceil(
        ideal_batch_size / training_args.batch_size
    ).astype(int)
    if training_args.verbose:
        logger.info(
            f"max optimiser steps {max_steps}",
        )
        logger.info(
            f"accumulate_grad_batches {accumulate_grad_batches}",
        )
        logger.info(
            f"accumulated batch size {training_args.batch_size * accumulate_grad_batches}"
        )
        logger.info(f"epochs {training_args.n_samples//len(dl_train.dataset)}")

    model_kwargs = {k: getattr(training_args, k) for k in training_args._model_keys}
    pl_model = PL_MODEL(
        model,
        adam8bit=training_args.load_in_4bit
        or training_args.load_in_8bit,  # saved mem, but seems unstable?
        schedule="wsd",
        num_iterations=max_steps,
        batch_size=training_args.batch_size,
        # model args
        **model_kwargs,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # checkpoint_callback
    ]
    if training_args.verbose:
        callbacks += [GenCallback(every=max_steps // 5 + 1)]

    model_kwargs = {k: getattr(training_args, k) for k in training_args._model_keys}
    trainer = pl.Trainer(
        max_steps=max_steps,
        limit_val_batches=6,
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
        fast_dev_run=training_args.dev,
        enable_progress_bar=training_args.verbose > 0,
        enable_model_summary=training_args.verbose > 1,
    )

    # train
    trainer.fit(pl_model, dl_train, dl_val)

    # save as regular adapter only

    model.save_pretrained(
        str(save_dir / "adapter"),
    )
    logger.info(f"saved to {save_dir/'adapter'}")

    # ### Hist
    if not training_args.dev:
        df_hist = (
            read_metrics_csv(trainer.logger.experiment.metrics_file_path)
            .bfill()
            .ffill()
        )
        # logger.info(df_hist)

    # eval(model, tokenizer, training_args, save_dir, run, finetune_name, adapter_name)

    # ## Gen
    model.cuda()  # for some reason it ends up cpu

    if (not training_args.dev) and (training_args.verbose > 0):
        df_gen = get_model_generations(model, tokenizer, N=3)
        display_gen(df_gen.head(2))

    logger.info("eval")

    # ## Eval
    # eval on ethics, GENIES, and our train dataset
    N = training_args.eval_samples
    datasets = [
        load_dataset_n(
            "wassname/genies_preferences",
            name=training_args.dataset,
            split="train",
            N=750 if N is None else min(N, 750),
        ),
        load_dataset_n(
            "wassname/genies_preferences", name=training_args.dataset, split="test", N=N
        ),
    ]
    datasets += dist2datasets(
        GENIES,
        # N=N, # can't cheap out on the main metric
        source=[training_args.dataset],
    )  # our hard OOS test
    # datasets += get_ethics_datasets(N=N)
    datasets += [
        load_dataset_n("wassname/genies_preferences", name=name, split="test", N=N)
        for name in [
            "math_make_questions",  #'truthful_qa',# 'wrong_arc', 'ranking_logic',
            # 'math', 'sycophancy_mimicry'
        ]
    ]

    clear_mem()
    res, df_res2 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_size=training_args.batch_size,
        bf16=True,
        torch_empty_cache_steps=200,
        verbose=training_args.verbose,
    )

    ds_alias = OrderedDict(
        list(zip(["train", "test", "oos", "rnd"], [ds2name(d) for d in datasets]))
    )

    # save
    f = str(save_dir) + "/eval.parquet"
    df_res2.to_parquet(f)
    logger.info(f"save_dir={save_dir}")
    pprint(training_args, compact=1)

    r = parse_eval(df_res2, ds_alias)

    # WANDB logging
    r2 = {}
    for k, v in r.items():
        r2[k] = wandb.Table(dataframe=v.reset_index())
        # log first row too, so we can compare single value
        run.log({f"res/{k}/{kk}": vv for kk, vv in v.iloc[0, :].to_dict().items()})
        # also just log final metrics to wandb so we can view a group

    # FIXME, only pass in adapter col, not q index or base
    if (not training_args.dev) and (training_args.verbose > 0):
        df_gen_w = wandb.Table(dataframe=df_gen)
        run.log({"generations": df_gen_w, **r2})

    if wandb.run is not None:
        logger.info(f"WANDB url = {wandb.run.get_url()}")

    # return a single dict for hyperparam tuning
    dd = [
        {f"{metric}/{split}": v for split, v in df.iloc[0].to_dict().items()}
        for metric, df in r.items()
    ]
    rd = {}
    [rd.update(**ddd) for ddd in dd]
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
    df_res_logp = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["_chosen_ppl"]
        .mean()
        .unstack()
        .T
    )
    perplexity_gain_vs_ref = df_res_logp.loc[adapter_name] / df_res_logp.loc["base"]

    # metric: how much does the model follow the preference vs the rejected. log ratio difference or ρθ in GPO https://arxiv.org/html/2402.05749v2
    df_res2["logratios"] = df_res2["_chosen_logps"] - df_res2["_rejected_logps"]
    df_logratios = (
        df_res2.groupby(["dataset", "adapter", "ds_i"])["logratios"].mean().unstack(0)
    )
    model_logratios = df_logratios.loc[adapter_name]
    ref_logratios = df_logratios.loc["base"]
    preference_logp_gain_vs_ref = (model_logratios - ref_logratios).mean()

    # metric: how much do we prefer the chosen over the rejected? This is `model_logratios` in DPO
    c = df_res_logp = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["_chosen_logps"]
        .mean()
        .unstack()
        .T.loc[adapter_name]
    )
    r = df_res_logp = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["_rejected_logps"]
        .mean()
        .unstack()
        .T.loc[adapter_name]
    )
    preference_logp_gain = c - r

    def fmt(s):
        return s.replace("genies_preferences-", "")

    def generate_metrics(metric_name, datasets, values):
        return [
            [metric_name, split, fmt(ds_name), values[ds_name]]
            for split, ds_name in datasets.items()
        ]

    # Define the datasets
    datasets = {
        "train": ds_name_train,
        "test": ds_name_test,
        "oos": ds_name_oos,
        "rnd": ds_name_rnd,
    }

    # Generate metrics for each category
    acc_gain_vs_ref_metrics = generate_metrics(
        "acc_gain_vs_ref", datasets, acc_gain_vs_ref
    )
    perplexity_gain_vs_ref_metrics = generate_metrics(
        "perplexity_gain_vs_ref", datasets, perplexity_gain_vs_ref
    )
    preference_logp_gain_metrics = generate_metrics(
        "preference_logp_gain", datasets, preference_logp_gain
    )
    preference_logp_gain_vs_ref_metrics = generate_metrics(
        "preference_logp_gain_vs_ref", datasets, preference_logp_gain_vs_ref
    )

    # Concatenate all metrics
    all_metrics = (
        acc_gain_vs_ref_metrics
        + perplexity_gain_vs_ref_metrics
        + preference_logp_gain_metrics
        + preference_logp_gain_vs_ref_metrics
    )

    # Create DataFrame
    df_metrics = pd.DataFrame(
        all_metrics, columns=["metric", "split", "dataset", "value"]
    )[["metric", "split", "value", "dataset"]]
    df_metrics = df_metrics.set_index(["metric", "split"])
    df_metrics = df_metrics["value"].unstack()
    df_metrics.index.name = f"{adapter_name}\dist shift"

    return df_metrics[list(ds_alias.keys())]


def parse_eval(df_res2, ds_alias):
    adapter_name = df_res2[["adapter"]].query('adapter!="base"').values[0, 0]

    df_res = (
        df_res2.groupby(["dataset", "adapter"], dropna=False)["correct"]
        .mean()
        .unstack()
        .T
    )
    df_res.columns = [d.replace("genies_preferences-", "") for d in df_res.columns]

    df_metrics = key_metrics(df_res2, adapter_name, ds_alias)

    logger.info(f"\n{df_metrics.round(3).to_markdown()}")
    logger.info("""Table 1: Key metrics (adapter over base model)\n""")

    cols = [v.replace("genies_preferences-", "") for v in ds_alias.values()]
    df_res2 = df_res[cols]
    df_res2.columns = list(ds_alias.keys())
    df_res2.index.name = "adapter/ds"
    logger.info(f"\n{df_res2.round(3).to_markdown()}")
    logger.info("""Table 2: Absolute accuracy\n""")

    df_final = df_metrics.loc["acc_gain_vs_ref"].to_frame(adapter_name).T
    df_final = df_final * 100 - 100  # percentage points
    df_final.index.name = "acc_inc/eval_ds [pp]"
    logger.info(f"\n{df_final.round(3).to_markdown()}")
    logger.info(
        f"""Table 3🥇: Accuracy increase (in percentage points) after training with named adapter on `{ds_alias["train"]}` compared to base model for various distribution shifts:"""
    )
    for k, v in ds_alias.items():
        logger.info(f"- `{k}`: `{v}`")

    df_acc = df_res2.loc[adapter_name].to_frame(adapter_name).T
    info = {
        "acc": df_acc,
        "acc_gain_vs_ref": df_final,
    }

    # format for wandb. just one row, one data type per table
    for i in df_metrics.index:
        info[i] = df_metrics.loc[i].to_frame(adapter_name).T
        info[i].index.name = adapter_name

    return info
