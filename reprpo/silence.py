import warnings, os
import logging
from datasets.utils.logging import disable_progress_bar


def remove_warnings():
    # warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    # warnings.filterwarnings("ignore", ".*divide by zero.*")
    warnings.filterwarnings(
        "ignore",
        ".*torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    )
    warnings.filterwarnings("ignore", ".*`do_sample` is set to.*")
    warnings.filterwarnings(
        "ignore",
        ".*None of the inputs have requires_grad=True. Gradients will be None.*",
    )

    warnings.filterwarnings(
        "ignore",
        ".*ou have modified the pretrained model configuration to control generation.*",
    )
    # https://github.com/huggingface/transformers/blob/14ee2326e51cb210cec72f31b248cb722e9d5d1f/src/transformers/models/phi3/modeling_phi3.py#L600
    warnings.filterwarnings(
        "ignore", ".*input hidden states seems to be silently casted in float32.*"
    )

    # https://github.com/huggingface/transformers/blob/3e96a0c32b7fcebdf8992e5ad8161272e4651618/src/transformers/trainer.py#L816
    warnings.filterwarnings(
        "ignore",
        ".*Trainer\.tokenizer is now deprecated.*",
    )
    # Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.


def silence():
    # wandb logger is too verbose
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    os.environ["WANDB_SILENT"] = "true"

    logging.getLogger("transformers.trainer").setLevel

    # datasets is too verbose
    disable_progress_bar()
    # from datasets.utils.logging import set_verbosity_error
    # set_verbosity_error()

    warnings.filterwarnings("ignore", category=UserWarning)

    # Silence all loggers with "transforms" in their name
    for name in logging.root.manager.loggerDict:
        blocklist = ["transformers", "lightning", "wandb"]
        if any(blocked in name for blocked in blocklist):
            logging.getLogger(name).setLevel(logging.ERROR)

    # logging.basicConfig(level=logging.WARNING)

    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def test():
    os.environ["DEBUG"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    # os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "true"
