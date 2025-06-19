from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from open_pref_eval.data import tokenize_dataset

class PrefDataModule(LightningDataModule):
    """DataModule for preference datasets using a central ExperimentConfig."""
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.ds = None

    def setup(self, stage=None):
        # allow different dataset paths using args.dataset
        if self.ds is not None:
            logger.warning("Dataset already loaded, skipping setup.")
            return
            
        # So we can have `wassname/medical-dpo-V2#data` or `wassname/ethics_expression_preferences:commonsense` so ds:subset#split
        if ('#' in self.args.dataset) or (':' in self.args.dataset):
            path, config_name = self.args.dataset.split('#', 1) if '#' in self.args.dataset else (self.args.dataset, None)
            path, split = path.split(':', 1) if ':' in path else (path, None)
        else:
            path = "wassname/genies_preferences"
            config_name = self.args.dataset
            split = None

        self.ds = load_dataset(
            path=path,
            name=config_name,
            split=split
        )

        tokenized_ds = tokenize_dataset(
            dataset=self.ds,
            tokenizer=self.tokenizer,
            max_prompt_length=self.args.max_prompt_length,
            max_length=self.args.max_length,
            verbose=self.args.verbose,
        )

        # remove truncated
        tokenized_ds = tokenized_ds.filter(
            lambda x: not (x['prompt_truncated'] or x['chosen_truncated'] or x['rejected_truncated']),
            desc="Filtering truncated samples"
        )


        if isinstance(tokenized_ds, dict):
            if 'train' in tokenized_ds and 'test' in tokenized_ds:
                pass
            else:
                tokenized_ds = next(iter(tokenized_ds.values())).train_test_split()
        else:
            tokenized_ds = tokenized_ds.train_test_split(test_size=0.1, seed=self.args.seed)
    
        self.train_ds = tokenized_ds['train']
        if len(self.train_ds) > self.args.n_samples:
            self.train_ds = self.train_ds.shuffle(seed=self.args.seed).select(range(self.args.n_samples))
            logger.info(f"Using {len(self.train_ds)} samples for training.")
        
        self.val_ds = tokenized_ds['test']
        if len(self.val_ds) > self.args.n_samples:
            self.val_ds = self.val_ds.select(range(self.args.n_samples))


    def train_dataloader(self):
        return DataLoader(
            self.train_ds.with_format("torch"),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds.with_format("torch"),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
