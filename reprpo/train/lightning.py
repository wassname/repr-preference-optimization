from torch import optim
import lightning as pl
from torchmetrics.functional import accuracy
import bitsandbytes as bnb

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

        self.log(f"{phase}/loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=self.hparams.batch_size)

        self.log_dict({
            f"{phase}/{k}": v for k, v in info.items()}, on_epoch=True, on_step=False, batch_size=self.hparams.batch_size),
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
            optimizer, self.hparams.lr, total_steps=self.hparams.num_iterations, verbose=True,
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
