# Instead of using the complex TRL we code it from scratch, using lighting
#
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

import json
import os
from pprint import pprint

from .silence import remove_warnings, silence

remove_warnings()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# from matplotlib import pyplot as plt
# lightning
import lightning as pl

# Numeric
import numpy as np
import pandas as pd
import torch
import transformers
import wandb
import yaml
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor

# get a union class from the enum
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from open_pref_eval.datasets import ds2name
from open_pref_eval.evaluation import evaluate_model
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial

# ML
from peft import LoraConfig, get_peft_model
from peft.tuners import LoraConfig
from torch.utils.data import DataLoader

from reprpo.data.collate3 import TokenizeRow

# Local
from reprpo.data.eval_sets import TRAINING_EXPERIMENTS, load_eval_ds
from reprpo.gen import display_gen, get_model_generations
from reprpo.helpers.lightning_hist import read_metrics_csv
from reprpo.helpers.pl_gen import GenCallback
from reprpo.helpers.torch import clear_mem
from reprpo.models.load import load_model, print_trainable_parameters

# LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
LOGURU_FORMAT = "<level>{message}</level>"
logger.remove()
logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="INFO")


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed to use (default: 42)
    """

    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)
    transformers.set_seed(seed)

    logger.info(f"Random seed set to {seed} for all libraries")


def apply_cfg_overrides(cfg, f=None):
    proj_root = Path(__file__).parent.parent
    overrides = {}
    if f is None:
        f = os.environ.get("REPR_CONFIG")
    if f is not None:
        f = proj_root / f
        logger.info(f"applying REPR_CONFIG {f}")
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                logger.warning(f"Warning: {k} not found in training_args")
        logger.info(f"loaded default config from {f}")
    return cfg


def flatten_dict(d, parent_key="", sep="."):
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

    diff = set(list2tuple(flatten_dict(asdict(args))).items()) - set(
        list2tuple(flatten_dict(asdict(defaults))).items()
    )
    diff = sorted(diff, key=lambda x: x[0])
    blacklist = [
        "eval_samples",
        "base_model",
        "dev",
        "verbose",
        "n_samples",
        "batch_size",
        "max_length",
        "max_prompt_length",
        "use_gradient_checkpointing",
        "load_in_4bit",
        "load_in_8bit",
        "collection_keys_in",
        "collection_keys_out",
        "collection_hs",
        "collection_layers",
        "collection_layers_hs",
        "save",
        "wandb",
    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2g}"
        return v

    s = " ".join([f"{k}={fmt(v)}" for k, v in list(diff) if k not in blacklist])

    cls_name = type(args).__name__
    if hasattr(args, "transform"):
        cls_name += f"_{type(args.transform).__name__.replace('Transform', '')}"
    if hasattr(args, "loss"):
        cls_name += f"_{type(args.loss).__name__.replace('Loss', '')}"
    cls_name = cls_name.replace("Config", "")

    # also either state transform and loss, or replace the words
    def rename(s):
        if hasattr(args, "loss"):
            loss_name = (
                type(args.loss)
                .__name__.lower()
                .replace("config", "")
                .replace("loss", "")
            )
            s = s.replace("loss.", f"{loss_name}.")
        if hasattr(args, "transform"):
            transform_name = type(args.transform).__name__.lower().replace("config", "")
            s = s.replace("transform.", f"{transform_name}.")
        return s

    s_all = " ".join([f"{k}={fmt(v)}" for k, v in list(diff)])
    s_short = f"{cls_name} {s}"
    s_all = rename(s_all)
    s_short = rename(s_short)
    logger.info(f"diff: {cls_name} {s_all}")

    return s_short


def nice_ds_name(s):
    """make a nice name for the dataset"""
    rm = [
        "genies_preferences-",
        "genies_",
        "genies-preferences-",
        "wassname/",
        "_expression_preferences",
    ]
    for r in rm:
        s = s.replace(r, "")

    # and remove "\[\:\d+\]" and replace with space
    import re

    s = re.sub(r"\[\:\d+\]", " ", s)

    # also remove -test or -data or -test-data or -train from end
    for suffix in ["-test", "-data", "-test-data", "-train"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]

    # replace
    rp = {
        "medica-dpo-v2": "cn-medical",
        "-": "_",
    }
    for k, v in rp.items():
        s = s.replace(k, v)

    return s


def safe_fn(s):
    """make a safe filename from any string."""
    return "".join(
        c for c in s if c.isalpha() or c.isdigit() or c == " " or c == "_" or c == "-"
    ).rstrip()


def train(args, trial: Optional[Trial] = None):
    if args.verbose < 1:
        silence()
    torch.set_float32_matmul_precision("medium")

    # Set random seed for reproducibility
    seed = getattr(args, "seed", 42)  # Default to 42 if no seed specified
    set_random_seed(seed)

    PL_MODEL = args._cls

    logger.info(f"PL_MODEL {PL_MODEL}")

    ds_name_train = args.dataset.replace("genies_preferences-", "")
    model_name = args.base_model.split("/")[-1]

    ts = pd.Timestamp.now().strftime("%H%M%S")
    adapter_name = args._name
    # adapter_name = get_args_dict(args)

    human_name = get_display_name_from_args(args)  # f"{adapter_name}_{ds_name_train}"

    # we can set an experiment group name from env vars, otherwise it will just groupt by model and training ds
    group_name = f"{ds_name_train}-{model_name}"
    if os.environ.get("WANDB_GROUP", None) is not None:
        group_name = safe_fn(os.environ.get("WANDB_GROUP") + "-" + group_name)

    if args.verbose > 1:
        logger.info("args")
        pprint(args, compact=True)
        # logger.info("model_kwargs", model_kwargs.keys())

        logger.info(f"Using finetune_name={human_name}")

    run_fname = f"{adapter_name}/{ts}"  # short for wandb

    # save_dir
    timestamp = safe_fn(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    root_dir = Path(__file__).parent.parent
    model_fname = safe_fn(
        "_".join([args.base_model.replace("/", "-"), adapter_name, ds_name_train])
    )
    save_dir = root_dir / "outputs" / group_name / f"{model_fname}" / f"{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    config = asdict(args)
    config.update(
        {
            "post": {
                "group_name": group_name,
                "adapter_name": adapter_name,
                "human_name": human_name,
                "model_fname": model_fname,
                "ds_name_train": ds_name_train,
                "run_fname": run_fname,
                "save_dir": str(save_dir),
                "ts": ts,
            }
        }
    )
    if args.wandb:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pl_wandb_logger = WandbLogger(
                name=run_fname,
                save_dir=save_dir,
                project="reprpo2",
                # entity="wassname",
                group=group_name,
                config=config,
                mode="disabled"
                if os.environ.get("WANDB_MODE", None) == "disabled"
                else "online",
            )

        # in case we already initialised it earlier, update it
        if wandb.run:
            wandb.config.update(config, allow_val_change=True)
            wandb.run.tags = tuple(wandb.run.tags) + (
                f"ds:{ds_name_train}",
                f"m:{model_fname}",
                f"i:{adapter_name}",
            )
            wandb.run.name = run_fname
            # wandb.run._group = group_name # can't change this
            wandb.run._quiet = True

            logger.info(
                f"📌Using WANDB_GROUP= https://wandb.ai/wassname/reprpo2/groups/{wandb.run.group} 📎"
            )
    # run = pl_wandb_logger._experiment

    # config
    (save_dir / "config.json").open("w").write(json.dumps(config, indent=4))

    # logging
    if args.verbose < 1:
        logger.remove()  # info by default
        logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="WARNING")
    log_file = save_dir / "log.txt"
    logger.add(
        log_file,
        level="INFO",
        format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"Logging to {log_file}")
    logger.info(f"Using save_dir={save_dir}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")

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
        target_modules=["q_proj", "v_proj"] if "qwen3" in args.base_model.lower() else None,
        # target_modules="all-linear", #  QLoRA-style training
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

    # ## Load training data
    ds_train = load_dataset("wassname/genies_preferences", name=args.dataset)
    ds_train_tok = ds_train.map(tokenize_row, batched=False)

    if args.verbose > 0:
        pt = np.mean(ds_train_tok["train"]["prompt_truncated"])
        ct = np.mean(ds_train_tok["train"]["chosen_truncated"])
        if pt > 0.2:
            logger.error(
                f"Prompt truncated {pt:2.2%} in {args.dataset} with {args.max_prompt_length} max length"
            )
        if ct > 0.2:
            logger.error(
                f"Chosens truncated {ct:2.2%} in {args.dataset} with {args.max_length} max length"
            )
        logger.info(f"Prompt truncated {pt:2.2%}")
        logger.info(f"Chosens truncated {ct:2.2%}")

        logger.info(
            f"Prompts truncated {np.mean(ds_train_tok['train']['prompt_truncated']):2.2%}"
        )
        logger.info(
            f"Chosens truncated {np.mean(ds_train_tok['train']['chosen_truncated']):2.2%}"
        )
        # FIXME in genies they filter out thos that are larger than max legnth https://github.com/Joshuaclymer/GENIES/blob/22c8afb2551851fb3f2d1a2dcf70e7608908f6b1/src/api/data_classes.py#L171

    def ds2dl(ds):
        return DataLoader(
            ds.select_columns(
                ["chosen", "rejected", "chosen_mask", "rejected_mask"]
            ).with_format("torch"),
            batch_size=args.batch_size,
        )

    dl_train = ds2dl(ds_train_tok["train"])
    dl_val = ds2dl(
        ds_train_tok["test"]
    )  # If we use stopping or reduce learnign on plateau, we need a validation set

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

    ideal_batch_size = max(
        16, args.batch_size
    )  # probobly wont be stable with less than 16, so make up the difference with gradient accumulation
    accumulate_grad_batches = np.ceil(ideal_batch_size / args.batch_size).astype(int)
    max_opt_steps = args.n_samples // (args.batch_size * accumulate_grad_batches)
    if args.verbose > 1:
        logger.info(
            f"max optimiser steps {max_opt_steps}",
        )
        logger.info(
            f"accumulate_grad_batches {accumulate_grad_batches}",
        )
        logger.info(f"effective batch size {args.batch_size * accumulate_grad_batches}")
        logger.info(f"epochs {args.n_samples / len(dl_train.dataset)}")

    model_kwargs = {k: getattr(args, k) for k in args._model_keys}
    pl_model = PL_MODEL(
        model,
        adam8bit=args.load_in_4bit
        or args.load_in_8bit,  # saved mem, but seems unstable?
        schedule="wsd",
        num_iterations=max_opt_steps,
        batch_size=args.batch_size,
        # model args
        **model_kwargs,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # checkpoint_callback
    ]
    if trial is not None:
        callbacks += [PyTorchLightningPruningCallback(trial, "val/dpo_acc")]
    if args.verbose > 1:
        callbacks += [GenCallback(every=max_opt_steps // 2 + 1)]

    model_kwargs = {k: getattr(args, k) for k in args._model_keys}
    loggers = [CSVLogger(name=run_fname, save_dir=save_dir, flush_logs_every_n_steps=5)]
    if args.wandb:
        loggers.append(pl_wandb_logger)
    trainer = pl.Trainer(
        max_steps=max_opt_steps,
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
        enable_progress_bar=args.verbose > 1,
        enable_model_summary=args.verbose > 1,
    )
    trainer.logger.log_hyperparams(config)

    # train
    try:
        trainer.fit(pl_model, dl_train, dl_val)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, stopping training")
        pass

    # save as regular adapter only (small)
    if args.save:
        model.save_pretrained(
            str(save_dir / "adapter"),
        )
        logger.info(f"saved to {save_dir / 'adapter'}")

    # ### Hist
    if not args.dev:
        df_hist = (
            read_metrics_csv(trainer.logger.experiment.metrics_file_path)
            .bfill()
            .ffill()
        )
        if args.verbose > 2:
            logger.info(df_hist)

    # eval(model, tokenizer, args, save_dir, run, finetune_name, adapter_name)

    # ## Gen
    model.cuda()  # for some reason it ends up cpu

    if (not args.dev) and (args.verbose > 0):
        N = args.verbose * 2
        df_gen = get_model_generations(model, tokenizer, N=N)
        display_gen(df_gen.head(N))

    logger.debug("eval")

    ds_train_tok = dl_train = dl_val = None
    trainer = None
    # ## Eval

    N = args.eval_samples
    ds_names_eval = TRAINING_EXPERIMENTS[args.dataset]
    df_ds_names_eval = pd.DataFrame(ds_names_eval)
    datasets = [load_eval_ds(name, N=N) for name in df_ds_names_eval["target"]]
    ds_names = [ds2name(d) for d in datasets]
    logger.info(f"evaluating on datasets: {ds_names}")

    remove_warnings()
    clear_mem()
    res, df_res2 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    df_res2.fillna({"adapter": "base"}, inplace=True)

    df_ds_names_eval["dataset"] = ds_names
    df_ds_names_eval["ds_name_nice"] = (
        df_ds_names_eval["type"]
        + " ("
        + df_ds_names_eval["dataset"].map(nice_ds_name)
        + ")"
    )  # +df_ds_names_eval['type_ds']#+'-'+df_res2['category_ds']
    df_res2 = df_res2.merge(
        df_ds_names_eval,
        on="dataset",
        suffixes=("", "_ds"),
        how="left",
    )

    # save
    f = str(save_dir) + "/eval.parquet"
    df_res2.to_parquet(f)
    logger.info(f"save_dir={save_dir}")
    # pprint(args, compact=1)

    r = make_table(df_res2, args, human_name=human_name, base_model=model_name)

    # WANDB logging
    r2 = {}
    for k, v in r.items():
        r2[k] = wandb.Table(dataframe=v.reset_index())
        # log first row too, so we can compare single value
        if wandb.run is not None:
            wandb.log(
                {f"res/{k}/{kk}": vv for kk, vv in v.iloc[0, :].to_dict().items()}
            )
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


def make_table(df_res2, args, human_name, base_model="", verbose=True):
    remove_warnings()
    adapter_name = df_res2[["adapter"]].query('adapter!="base"').values[0, 0]

    df_res_ds = (
        df_res2.groupby(["ds_name_nice", "adapter"], dropna=False)["correct"]
        .mean()
        .unstack()
        .T
    )
    df_res_ds.index.name = "adapter/ds"

    df_res_type = (
        df_res2.groupby(["type", "adapter"], dropna=False)["correct"].mean().unstack().T
    )
    df_res_type.index.name = "adapter/distribution_shift"

    # TODO order it so in_domain is first, unrelated is last
    cols = df_res_type.columns
    cols = (
        ["in_domain"]
        + [c for c in cols if c not in ["in_domain", "control"]]
        + ["control"]
    )
    df_res_type = df_res_type[cols]

    # this was getting ppx, logratio, nll (from _chosen_ppl/_rejected_ppl,etc)

    # df_res = df_res[list(ds_alias.keys())]

    caption = f"""Table 1: Absolute accuracy after training with named adapter compared to base model `{base_model}` for various distribution shifts [N={args.eval_samples}]:\n"""
    x = df_res2.groupby("type")["dataset"].agg(lambda s: set(s)).to_dict()
    for k, v in x.items():
        caption += f"- Shift: {k}, made up of:\n"
        for vv in v:
            caption += f"\t- `{vv}`\n"
    logger.info(f"\n{df_res_type.round(3).to_markdown()}")
    logger.info(caption)

    caption = f"""Table 2: Absolute accuracy after training with named adapter on ds:`{args.dataset}` compared to base model `{base_model}` for various distribution shifts [N={args.eval_samples}]:\n"""
    logger.info(f"\n{df_res_ds.round(3).to_markdown()}")
    logger.info(caption)

    df_adapter_acc = df_res_ds.loc[adapter_name].to_frame(human_name).T
    wandb_info = {
        "acc": df_adapter_acc,
        # "acc_gain_vs_ref": df_final,
    }

    return wandb_info
