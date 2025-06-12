from pathlib import Path
from loguru import logger
import os
import yaml
from dataclasses import asdict, dataclass

def apply_cfg_overrides_from_env_var(cfg, f=None):
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
    """
    extract a human readable name from non-default args
    TODO make this unit tested or tidier
    """
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

    # don't show some keys
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
        "lr",
        "weight_decay",
        'eps',
        'collect_hs', # covered in name
        # 'α',
        'seed',

    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2g}"
        return v

    
    # filter
    diff2 = []
    for k, v in diff:
        if k in blacklist:
            continue
        if any(
            k.endswith(bl) for bl in blacklist
        ):
            continue
        diff2.append((k, v))



    cls_name = type(args).__name__
    if hasattr(args, "transform"):
        cls_name += f"_{type(args.transform).__name__.replace('Transform', '')}"
    if hasattr(args, "loss"):
        cls_name += f"_{type(args.loss).__name__.replace('Loss', '')}"
    cls_name = cls_name.replace("Config", "")
    s_short = " ".join([f"{k}={fmt(v)}" for k, v in diff2])
    s_all = " ".join([f"{k}={fmt(v)}" for k, v in list(diff)])

    # also either state transform and loss, or replace the words
    def rename(s):
        """
        loss. -> innerdpo
        transform -> ether
        """
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

    
    
    # s_short = rename(s_short)
    
    s_all = rename(s_all)
    logger.info(f"diff: {cls_name} {s_all}")


    def acronym(s): 
        defs = {
            'True': '1',
            'False': '0',
            'ReprPO': 'Repr',
            'InnerDPO': 'IPO',
            'None': 'n',
            'loss.': 'l.',
            'transform.': 't.',
            "alpha": 'α',
            "beta": 'β',
            "gamma": 'γ',
            "epsilon": 'ε',
            "eps": 'ε',
        }
        for k, v in defs.items():
            s = s.replace(k, v)
            s = s.replace(k.lower(), v)
        return s

    def snake_case_acronym(k, keep=2, sep=''):
        k = k.replace('-', '_')
        kl = [ll.capitalize()[:keep] for ll in k.split('_')]
        return sep.join(kl)

    def make_shorter(s):
        """
        e.g. ReprPO_ETH collect_hs=True innerdpo.align_method=orth innerdpo.eps=1e-05 innerdpo.norm_before_reduce=True innerdpo.use_policy_weights=True innerdpo.α=1 verbose=2
        to ReprPO_ETH CoHs=1 AlMe=orth eps=1e-05 NoBeR=1 UsPoW=1 α=1
        """
        # rm everything after the first dot
        sl = [ss.split('.', 1)[-1] for ss in s.split()]
        # now for the part before an =
        sl2 = []
        for ss in sl:
            if '=' in ss:
                k, v = ss.split('=', 1)
                if len(k) >5:
                    # take the first letter of underscored or dash keys
                    k = snake_case_acronym(k, keep=3)
                if len(v) > 7:
                    v = snake_case_acronym(v, keep=5)
                k = k[:7]
                v = v[:7]
                if k not in blacklist:
                    sl2.append(f"{k}={v}")
            else:
                sl2.append(ss[:10])  # take the first 5 chars

        return " ".join(sl2)
    

    cls_name_shorter = snake_case_acronym(acronym(cls_name), keep=4, sep='')
    shorter2 = make_shorter(acronym(s_short))
    shorter = f"{cls_name_shorter} {shorter2}"

    cls_name_short = snake_case_acronym(acronym(cls_name), keep=5, sep='')
    s_short2 = acronym(f"{cls_name_short} {s_short}")
    return s_all, s_short2, shorter
