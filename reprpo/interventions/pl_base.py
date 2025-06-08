from torch import optim
import lightning as pl
import bitsandbytes as bnb
from dataclasses import dataclass
from transformers.optimization import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_inverse_sqrt_schedule, get_wsd_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

# @dataclass
# class ModelConfigBase:
#     lr: float = 3e-4
#     weight_decay: float = 0.0


class PL_MODEL(pl.LightningModule):
    _modules = {}  # HACK for jaxtyping

    def __init__(
        self,
        model,
        num_iterations,
        lr=None,
        weight_decay=0,
        batch_size=None,
        adam8bit=False,
        schedule="warmup_cosine",
        use_grad_paging=False,
    ):
        super().__init__()
        self._model = model
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self._model(x)

    def _loss_fn(self, batch, model):
        raise NotImplementedError("Implement in subclass")

    def _shared_step(self, batch, batch_idx, phase="train"):
        loss, info = self._loss_fn(batch, self._model)

        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )

        # also consider logging nll and dpo_acc
        self.log(
            f"{phase}/dpo_acc",
            info.pop("_dpo_acc"),
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            f"{phase}/nll_rat",
            info.pop("_nll_lratio"),
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )

        self.log_dict(
            {f"{phase}/{k}": v for k, v in info.items()},
            # on_epoch=True,
            on_step=True,
            batch_size=self.hparams.batch_size,
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
        elif self.hparams.use_grad_paging:
            # paged optimizer avoid memory spikes  https://arxiv.org/pdf/2305.14314
            optimizer = bnb.optim.PagedAdam32bit(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            # normal adamw, safest
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # best according to https://arxiv.org/pdf/2107.04197
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.num_iterations * 0.1),
            num_training_steps=self.hparams.num_iterations,
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]
