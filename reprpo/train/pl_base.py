from torch import optim
import lightning as pl
from torchmetrics.functional import accuracy
import bitsandbytes as bnb
import torch
from dataclasses import dataclass
from ..helpers.scheduler import get_constant_schedule_with_warmup


@dataclass(frozen=True)
class TrainingArguments():
    
    """the dataset to fine tune on. see subsets in https://huggingface.co/datasets/wassname/genie_dpo"""
    dataset: str = 'us_history_textbook'

    verbose: bool = False

    """fast run"""
    dev: bool = False

    load_in_4bit: bool = False  # this doesn't seem to be able to backprop when using baukit
    load_in_8bit: bool = False  # this doesn't seem to be able to backprop when using baukit
    use_gradient_checkpointing: bool = False

    # train
    batch_size: int = 8
    lr: float = 6e-5
    weight_decay: float = 0.0

    # dataset
    n_samples: int = 1800 * 3
    max_length: int = 196
    max_prompt_length: int = 96

    # model
    # model_name: str = "microsoft/Phi-3-mini-4k-instruct" # small instruct model
    # model_name: str = "google/gemma-2-2b" # small base model
    # model_name: str = "NousResearch/Meta-Llama-3.1-8B" # med base model
    # model_name: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    # collection_layers: tuple=(10, 12, 14, 16, 18, 20, 22, 24, 26, 28) 

    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    collection_layers: tuple=(10, 12, 14, 16, 18) 
    collection_keys_in: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    )
    collection_keys_out: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.q_proj",
        "base_model.model.model.layers.{layer}.self_attn.k_proj",
        "base_model.model.model.layers.{layer}.self_attn.v_proj",
        "base_model.model.model.layers.{layer}.mlp.gate_proj",
        "base_model.model.model.layers.{layer}.mlp.up_proj",
    )


class PL_MODEL(pl.LightningModule):
    def __init__(self, model, num_iterations, lr=3e-4, weight_decay=0, batch_size=None, adam8bit=False, schedule='cosine'):
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
            num_warmup_steps = self.hparams.num_iterations * 0.03
            scheduler= get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]


from reprpo.gen import get_model_generations
from lightning.pytorch.callbacks import Callback


class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=50):
        self.every = every

    def do_gen(self, model):
        get_model_generations(model, model.tokenizer, max_new_tokens=32, N=1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if n % self.every == 0:
            print(f"Generated on batch {batch_idx}")
            self.do_gen(trainer.model._model)

    def on_train_epoch_end(self, trainer, pl_module):
        self.do_gen(trainer.model._model)


def cross_entropy_loss(logits, labels):
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)
    return loss
