from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from open_pref_eval.trainer import DataCollatorForPreference, tokenize_dataset

class PrefDataModule(LightningDataModule):
    """DataModule for preference datasets using a central ExperimentConfig."""
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        # allow different dataset paths using args.dataset
        self.ds = load_dataset(
            *(self.args.dataset.split(":", 1))
        ) if ":" in self.args.dataset else load_dataset(
            "wassname/genies_preferences", name=self.args.dataset
        )

        tokenized_ds = tokenize_dataset(
            dataset=self.ds,
            tokenizer=self.tokenizer,
            max_prompt_length=self.args.max_prompt_length,
            max_length=self.args.max_length,
            verbose=self.args.verbose,
        )
        
        # tok = self.ds.map(self.tokenize, batched=False)
        self.train_ds = tokenized_ds['train']
        self.val_ds = tokenized_ds['test']

        self.data_collator = DataCollatorForPreference(
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
        )

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
