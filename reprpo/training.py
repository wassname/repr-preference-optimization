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
from loguru import logger
from open_pref_eval.datasets import ds2name
from open_pref_eval.evaluation import evaluate_model
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial

# ML
from peft import LoraConfig, get_peft_model
from peft.tuners import LoraConfig

from reprpo.data.datamodule import PrefDataModule  # noqa: E402

# Local imports (noqa for import placement)
from reprpo.data.eval_sets import TRAINING_EXPERIMENTS, load_eval_ds  # noqa: E402
from reprpo.gen import display_gen, get_model_generations  # noqa: E402
from reprpo.helpers.lightning_hist import read_metrics_csv  # noqa: E402
from reprpo.helpers.pl_gen import GenCallback  # noqa: E402
from reprpo.helpers.torch import clear_mem  # noqa: E402
from reprpo.models.load import load_model, print_trainable_parameters  # noqa: E402
from reprpo.helpers.logging import setup_logging  # centralized log setup
from reprpo.helpers.wandb_utils import init_wandb  # noqa: E402
from reprpo.helpers.tyro import get_display_name_from_args, apply_cfg_overrides
from reprpo.data.util import nice_ds_name, safe_fn  # noqa: E402

# LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
LOGURU_FORMAT = "<level>{message}</level>"
logger.remove()
logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="INFO")


def set_random_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    transformers.set_seed(seed)
    logger.info(f"Random seed set to {seed} for all libraries")


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
    # centralized logging setup
    setup_logging(str(save_dir), level="DEBUG" if args.verbose > 1 else "INFO")
    logger.info(f"Logging initialized at {save_dir}")

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
        pl_wandb_logger = init_wandb(
            args, str(save_dir), group_name, run_fname, project="reprpo2"
        )
    # run = pl_wandb_logger._experiment

    # config
    (save_dir / "config.json").open("w").write(json.dumps(config, indent=4))

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

    # setup data with PrefDataModule (replaces manual load/map/dl)
    dm = PrefDataModule(args, tokenizer)
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
        logger.info(f"epochs {args.n_samples / len(dm.train_dataloader().dataset)}")

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
        trainer.fit(pl_model, datamodule=dm)
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


    # ## Gen
    model.cuda()  # for some reason it ends up cpu

    if (not args.dev) and (args.verbose > 0):
        N = args.verbose * 2
        df_gen = get_model_generations(model, tokenizer, N=N)
        display_gen(df_gen.head(N))

    logger.debug("eval")

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
        + [c for c in cols if c not in ["in_domain", "orthogonal"]]
        + ["orthogonal"]
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
