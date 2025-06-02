from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader
from .collate3 import TokenizeRow
import numpy as np
from loguru import logger

class PrefDataModule(LightningDataModule):
    """DataModule for preference datasets using a central ExperimentConfig."""
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.tokenize = TokenizeRow(
            tokenizer,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        )

    def setup(self, stage=None):
        # allow different dataset paths using args.dataset
        self.ds = load_dataset(
            *(self.args.dataset.split(":", 1))
        ) if ":" in self.args.dataset else load_dataset(
            "wassname/genies_preferences", name=self.args.dataset
        )
        tok = self.ds.map(self.tokenize, batched=False)
        self.train_ds = tok['train']
        self.val_ds = tok['test']
        # QC: report truncation rates
        if getattr(self.args, 'verbose', 0) > 0:
            pt = np.mean(self.train_ds['prompt_truncated'])
            ct = np.mean(self.train_ds['chosen_truncated'])
            if pt > 0.2:
                logger.error(f"Prompt truncated {pt:2.2%} > {self.args.max_prompt_length}")
            if ct > 0.2:
                logger.error(f"Chosen truncated {ct:2.2%} > {self.args.max_length}")
            logger.info(f"Prompts truncated {pt:2.2%}, Chosen truncated {ct:2.2%}")
        # QC sample decode when verbose
        if getattr(self.args, 'verbose', 0) > 2:
            dl = DataLoader(self.train_ds.with_format('torch'), batch_size=self.args.batch_size)
            batch = next(iter(dl))
            logger.info("=== Sample QC after tokenization ===")
            logger.info(self.tokenizer.decode(batch['chosen'][0]))
            logger.info("---")
            logger.info(self.tokenizer.decode(batch['rejected'][0]))
            logger.info("=== End QC sample ===")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds.with_format("torch"),
            batch_size=self.args.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds.with_format("torch"),
            batch_size=self.args.batch_size,
        )
