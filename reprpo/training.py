# Instead of using the complex TRL we code it from scratch, using lighting
#
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

import json
import os
from pprint import pprint, pformat
from .silence import remove_warnings, silence
import datasets



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
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, BatchSizeFinder, LearningRateFinder

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
from reprpo.eval.gen import display_gen, get_model_generations  # noqa: E402
from reprpo.lightning.lightning_hist import read_metrics_csv  # noqa: E402
from reprpo.lightning.pl_gen import GenCallback  # noqa: E402
from reprpo.helpers.torch import clear_mem  # noqa: E402
from reprpo.models.load import load_model, print_trainable_parameters  # noqa: E402
from reprpo.helpers.logging import setup_logging  # centralized log setup
from reprpo.helpers.wandb_utils import init_wandb, flatten_dict  # noqa: E402
from reprpo.helpers.tyro import get_display_name_from_args, apply_cfg_overrides_from_env_var
from reprpo.data.util import nice_ds_name, safe_fn, df_sort_cols  # noqa: E402

# LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
LOGURU_FORMAT = "<level>{message}</level>"
logger.remove()
logger.add(os.sys.stderr, format=LOGURU_FORMAT, level="INFO")

import logging
logging.captureWarnings(True)


def set_random_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    transformers.set_seed(seed)
    logger.info(f"Random seed set to {seed} for all libraries")


def train(args, trial: Optional[Trial] = None):
    if args.dev:
        # no cache
        datasets.disable_caching()
        args.eval_samples = 32
        os.environ["WANDB_MODE"] = "disabled"  # disable wandb

    if args.verbose < 1:
        silence()
    if args.verbose < 3:
        remove_warnings()
    torch.set_float32_matmul_precision("medium")

    # Set random seed for reproducible
    seed = getattr(args, "seed", 42)  # Default to 42 if no seed specified
    set_random_seed(seed)

    
    PL_MODEL = args._cls

    logger.info(f"PL_MODEL {PL_MODEL}")

    ds_name_train = args.dataset.replace("genies_preferences-", "")
    model_name = args.base_model.split("/")[-1]

    ts = pd.Timestamp.now().strftime("%H%M%S")
    long_name, human_name, short_name = get_display_name_from_args(args)  # f"{adapter_name}_{ds_name_train}"
    adapter_name = args._name # pytorch module name, so cannot have a dot 
    # adapter_name = get_args_dict(args)
    


    # we can set an experiment group name from env vars, otherwise it will just groupt by model and training ds
    group_name = f"{ds_name_train}-{model_name}"
    if os.environ.get("WANDB_GROUP", None) is not None:
        group_name = safe_fn(os.environ.get("WANDB_GROUP") + "-" + group_name)

    short_human_name = short_name.split(' ', 1)[-1][:80] # safe_fn ?
    run_fname = f"{adapter_name}/{short_human_name}/{ts}"  # short for wandb

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
                "short_name": short_name,
                "long_name": long_name,
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
            config, str(save_dir), group_name, run_fname, project="reprpo2"
        )

    if args.verbose > 1:
        logger.info(f"args {pformat(args, compact=True)}")
        
        # logger.info("model_kwargs", model_kwargs.keys())

        logger.info(f"Using finetune_name={human_name}")
    # run = pl_wandb_logger._experiment

    # config
    (save_dir / "config.json").open("w").write(json.dumps(config, indent=4))

    model, tokenizer = load_model(
        args.base_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation='flash_attention_2',  # for gemma
        device_map='auto',
    )


    # ### Load adapter
    """
    Note that GENIES and PEFT use the default targets for llama ["q_proj", "v_proj"]
    but other papers like qlora and HRA use all linear layers
    """
    target_modules = None
    if "qwen3" in args.base_model.lower():
        target_modules = ["q_proj", "v_proj"]
    elif "OMLo" in args.base_model.lower():
        target_modules=["q_proj",  "v_proj", ]
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        use_rslora=True, # Rank-Stabilized LoRA
        # use_dora=True,
        task_type="CAUSAL_LM",
        # target_modules=target_modules,
        target_modules="all-linear", #  QLoRA-style training
    )

    model = get_peft_model(model, peft_config, adapter_name=adapter_name)
    print_trainable_parameters(model)
    logger.info(f"Using adapter `{adapter_name}` with target modules {peft_config.target_modules}")
    if args.verbose > 2:
        logger.info(f"{model}")
    config.update(
        {
            "peft_config": model.peft_config[adapter_name].to_dict() if hasattr(model, "peft_config") else None,
    })

    # setup data with PrefDataModule (replaces manual load/map/dl)
    dm = PrefDataModule(args, tokenizer)
    dm.setup()
    if wandb.run is not None:
        logger.info(f"WANDB url = {wandb.run.get_url()}")

    # ## Trainer
    # - https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/text-transformers.html
    # - https://gist.github.com/wassname/e29d02b5026a531e13912cf768e6fdc8

    ideal_batch_size = max(
        args.ideal_batch_size, args.batch_size
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
        # schedule=args.schedule,
        num_iterations=max_opt_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_grad_paging=args.use_grad_paging,
        # model args
        **model_kwargs,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # checkpoint_callback
    ]
    # if trial is not None:
    #     callbacks += [PyTorchLightningPruningCallback(trial, "val/dpo_acc")]
    # if args.verbose > 1:
    callbacks += [
        GenCallback(every=max_opt_steps // 2 + 1),
        EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
            check_finite=True,
            strict=False,
            verbose=True,
        ),
    ]

    model_kwargs = {k: getattr(args, k) for k in args._model_keys}
    loggers = [CSVLogger(name=run_fname, save_dir=save_dir, flush_logs_every_n_steps=5)]
    if args.wandb:
        loggers.append(pl_wandb_logger)
    trainer = pl.Trainer(
        max_steps=max_opt_steps,
        # limit_val_batches=max(6, max_steps//10),
        gradient_clip_val=args.gradient_clip_val,
        accelerator='gpu',
        # devices=1,
        # plugins=plugins,
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html
        # mixed wont convert the modules weights, while bf16-true would
        precision=args.pl_precision,
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
    trainer.logger.log_hyperparams(flatten_dict(config))

    # TODO consider learning rate finder
    if args.verbose > 2:
        from lightning.pytorch.callbacks import LearningRateFinder

    # train
    try:
        trainer.fit(pl_model, datamodule=dm)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, stopping training")
        pass

    # trainer ends by moving it to cpu, undo that
    model = model.to("cuda").to(torch.bfloat16)

    # save as regular adapter only (small)
    if args.save:
        model.save_pretrained(
            str(save_dir / "adapter"),
        )
        logger.info(f"saved to {save_dir / 'adapter' / adapter_name}")

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
    # model.cuda()  # for some reason it ends up cpu

    if (not args.dev) and (args.verbose > 0):
        N = min(2, args.verbose * 2)
        df_gen = get_model_generations(model, tokenizer, N=4)
        display_gen(df_gen.head(N))

    logger.debug("eval")

    trainer = None
    # ## Eval

    N = args.eval_samples
    ds_names_eval = TRAINING_EXPERIMENTS[args.dataset]
    df_ds_names_eval = pd.DataFrame(ds_names_eval)
    datasets_l = [load_eval_ds(name, N=N) for name in df_ds_names_eval["target"]]
    ds_names = [ds2name(d) for d in datasets_l]
    logger.info(f"evaluating on datasets: {ds_names}")

    
    clear_mem()
    res, df_res2 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets_l,
        batch_size=args.batch_size*2,
        verbose=args.verbose,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_workers=args.num_workers,
    )
    df_res2.fillna({"adapter": "base"}, inplace=True)
    df_res2['seed'] = seed
    df_res2['train'] = ds_name_train

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
    logger.info(f"""- save_dir={save_dir / "adapter" / adapter_name}
- Config: {pformat(config, compact=True)}
- Long name: {long_name}
- Human name: {human_name}
- Short name: {short_name}
- WANDB url = {wandb.run.get_url() if wandb.run is not None else 'None'})
""")

    r = make_table(df_res2, args, human_name=human_name, short_name=short_name, base_model=model_name)
    # and wandb url

    # WANDB logging
    wandb_tables = {}
    for k, v in r.items():
        wandb_tables[k] = wandb.Table(dataframe=v.reset_index())
        # log first row too, so we can compare single value
        if wandb.run is not None:
            wandb.log(
                {f"res/{k}/{kk}": vv for kk, vv in v.iloc[0, :].to_dict().items()}
            )
        # also just log final metrics to wandb so we can view a group

    if wandb.run is not None:
        if (not args.dev) and (args.verbose > 0):
            df_gen_w = wandb.Table(dataframe=df_gen)
            wandb.log({"generations": df_gen_w, **wandb_tables})

        

    # return a single dict for hyperparam tuning
    dd = [
        {f"{metric}/{split}": v for split, v in df.iloc[0].to_dict().items()}
        for metric, df in r.items()
    ]
    rd = {}
    [rd.update(**ddd) for ddd in dd]

    wandb.finish(quiet=True)
    return rd


def make_table(df_res2, args, human_name, base_model="", short_name="", verbose=True):
    
    adapter_name = df_res2[["adapter"]].query('adapter!="base" & adapter!="none"').values[0, 0]

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

    df_res_type = df_sort_cols(df_res_type,
        first=["in_domain"],
        last=["orthogonal"],
    )
    df_res_type = df_sort_cols(
        df_res_type.T,
        first=["base", "none"],
        last=[adapter_name],
    ).T
    df_res_type.index.name = "adapter/distribution_shift"



    caption = f"""Table 1: Absolute accuracy after training with named adapter on ds:`{args.dataset}` compared to base model `{base_model}` for various distribution shifts [N={args.eval_samples}]:\n"""
    x = df_res2.groupby("type")["dataset"].agg(lambda s: set(s)).to_dict()
    for k, v in x.items():
        caption += f"- Shift: {k}, made up of:\n"
        for vv in v:
            caption += f"\t- `{vv}`\n"
    logger.info(f"\n{df_res_type.round(3).to_markdown()}")
    logger.info(caption)




    caption = f"""Table 2: Absolute accuracy after training with named adapter on ds:`{args.dataset}` compared to base model `{base_model}` for various distribution shifts [N={args.eval_samples}]:\n"""
    logger.info(f"\n{df_res_ds.T.round(3).to_markdown()}")

    # also make a nice line for our records, with url, cli, and nll (coherency proxy)
    df_res_type['wandb'] = wandb.run.get_url().split('/')[-1] if wandb.run is not None else 'None'
    df_res_type.loc['none', 'wandb'] = None
    # df_res_type['short_name'] = short_name
    nll = df_res2.groupby('adapter')['_chosen_ppl'].apply(lambda x:np.log(x).mean())
    nll_rat = (nll.loc[adapter_name] - nll.drop(adapter_name).mean())

    # to measure coherency, we look at how the negative log likelihood of the chosen samples compares to the reference (base model). Generally [0-2] is ok, and we do expect some higher number from DPO in general, but >2 is likely to be incoherent
    df_res_type['nll_cho/ref'] = nll_rat
    record_line = df_res_type.loc[[adapter_name]]
    record_line.index = [short_name]
    logger.info(f"Record entry:\n\n{record_line.round(3).to_markdown()}\n")

    df_adapter_acc = df_res_ds.loc[adapter_name].to_frame(human_name).T
    wandb_info = {
        "acc": df_adapter_acc,
        # "acc_gain_vs_ref": df_final,
    }

    return wandb_info
