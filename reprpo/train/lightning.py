from torch import optim
import lightning as pl
from torchmetrics.functional import accuracy
import bitsandbytes as bnb

from dataclasses import dataclass

@dataclass
class TrainingArguments:
    # model
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    use_bnb: bool = True # this doesn't seem to be able to backprop when using baukit
    use_gradient_checkpointing: bool = False
    use_inputs: bool = True

    # train
    n_epochs: int = 1
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0

    # dataset
    n_samples: int = 1500 * 3 * 3
    max_length: int = 128
    max_prompt_length: int=64

class PL_MODEL(pl.LightningModule):
    def __init__(self, model, num_iterations, lr=3e-4, weight_decay=0, batch_size=None):
        super().__init__()
        self._model = model
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        return self._model(x)
    
    def _loss_fn(self, batch, model):
        raise NotImplementedError("Implement in subclass")

    def _shared_step(self, batch, batch_idx, phase='train'):        

        loss, info = self._loss_fn(batch, self._model)

        # self.log(f"{phase}/loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=self.hparams.batch_size)

        self.log_dict({
            f"{phase}/{k}": v for k, v in info.items()}, on_epoch=True, on_step=True, batch_size=self.hparams.batch_size),
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase='val')
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, batch_idx, phase='test')
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, batch_idx, phase='pred').float().cpu().detach()
    
    def configure_optimizers(self):
        """simple vanilla torch optim"""
        # https://lightning.ai/docs/fabric/stable/fundamentals/precision.html
        optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.lr, total_steps=self.hparams.num_iterations, verbose=True, pct_start=0.1,
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]

from reprpo.gen import get_model_generations
from lightning.pytorch.callbacks import Callback

class GenCallback(Callback):
    """A callback to generate on sample each N samples."""
    def __init__(self, every=50):
        self.every = every

    def do_gen(self):
        get_model_generations(self._model, self._model.tokenizer, N=1)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx%self.every==0:
            self.do_gen()

    def on_train_epoch_end(self, trainer, pl_module):
        get_model_generations(self._model, self._model.tokenizer, N=1)
