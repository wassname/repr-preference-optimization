from torch import optim
import lightning as pl
from torchmetrics.functional import accuracy
import bitsandbytes as bnb
import torch
from typing import Optional
from dataclasses import dataclass
from ..helpers.scheduler import get_constant_schedule_with_warmup


@dataclass
class TrainingArguments:
    
    """Fine tune dataset. see subsets in https://huggingface.co/datasets/wassname/genie_dpo"""

    dataset: str = 'us_history_textbook'
    """train dataset."""

    verbose: bool = False

    dev: bool = False
    """fast run"""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = False

    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.0

    n_samples: int = 1800 * 3
    eval_samples: Optional[int] = None
    max_length: int = 196
    max_prompt_length: int = 96
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class TrainingArgumentswCollection(TrainingArguments):
    alpha: float = 0.1

    collection_layers_side: tuple=(10, 12, 14, 16, 18) 
    """layers to collect activations from in side layers."""

    collection_layers_hs: tuple=(10, 20, 30)
    """The layers to collect the hidden states from. Thisis for methods that operate on the redundant residual stream so only needs a couple of points of collection"""
    
    collection_keys_in: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    )
    """keys to collect inputs from."""

    collection_keys_out: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.q_proj",
        "base_model.model.model.layers.{layer}.self_attn.k_proj",
        "base_model.model.model.layers.{layer}.self_attn.v_proj",
        "base_model.model.model.layers.{layer}.mlp.gate_proj",
        "base_model.model.model.layers.{layer}.mlp.up_proj",
    )
    """keys to collect outputs from."""

    collect_input: bool = True
    """use collection_keys_in? else use collection_keys_out."""


class PL_MODEL(pl.LightningModule):
    _modules = {} # HACK for jaxtyping

    def __init__(self, model, num_iterations, lr=3e-4, weight_decay=0, batch_size=None, adam8bit=False, schedule='constant'):
        super().__init__()
        self._model = model
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self._model(x)

    def _loss_fn(self, batch, model):
        raise NotImplementedError("Implement in subclass")

    def _shared_step(self, batch, batch_idx, phase="train"):
        loss, info = self._loss_fn(batch, self._model)

        self.log(f"{phase}/loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=self.hparams.batch_size)

        (
            self.log_dict(
                {f"{phase}/{k}": v for k, v in info.items()},
                on_epoch=True,
                on_step=True,
                batch_size=self.hparams.batch_size,
            ),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, batch_idx, phase="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, batch_idx, phase="pred").float().cpu().detach()

    def configure_optimizers(self):
        """simple vanilla torch optim"""
        # https://lightning.ai/docs/fabric/stable/fundamentals/precision.html
        if self.hparams.adam8bit:
            optimizer = bnb.optim.AdamW8bit(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            # paged optimizer avoid memory spikes  https://arxiv.org/pdf/2305.14314
            optimizer = bnb.optim.PagedAdam32bit(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # else:
        #     optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.schedule == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_iterations, eta_min=0)
        elif self.hparams.schedule == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.hparams.lr,
                total_steps=self.hparams.num_iterations,
                verbose=True,
                pct_start=0.1,
            )
        elif self.hparams.schedule == 'constant':
            # same as GENIES warmup_ratio=0.03
            num_warmup_steps = int(self.hparams.num_iterations * 0.03)
            scheduler= get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]


from reprpo.gen import get_model_generations, display_gen
from lightning.pytorch.callbacks import Callback


class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=50):
        self.every = every

    def do_gen(self, model):
        df_gen = get_model_generations(model, model.tokenizer, max_new_tokens=32, N=1)
        display_gen(df_gen, with_q=False)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if n % self.every == 0:
            print(f"\nGenerated on batch {batch_idx}")
            self.do_gen(trainer.model._model)

    def on_train_epoch_end(self, trainer, pl_module):
        self.do_gen(trainer.model._model)


def cross_entropy_loss(logits, labels, attn):
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    logits2 = logits.view(-1, logits.shape[-1])
    labels2 = labels.view(-1)
    # Enable model parallelism
    labels2 = labels2.to(logits.device)
    loss = loss_fct(logits2, labels2).view_as(labels) * attn.detach()
    return loss
