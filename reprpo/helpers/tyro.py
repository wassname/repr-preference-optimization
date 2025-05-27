from pathlib import Path
from loguru import logger
import os
import yaml
from dataclasses import asdict, dataclass

def apply_cfg_overrides(cfg, f=None):
    """
    Load a yaml, and override the tyro cfg
    """
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


def get_display_name_from_args(args: dataclass):
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
