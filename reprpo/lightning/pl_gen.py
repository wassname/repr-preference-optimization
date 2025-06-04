from reprpo.eval.gen import get_model_generations, display_gen
from lightning.pytorch.callbacks import Callback
from loguru import logger
import wandb

class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=50):
        self.every = every
        self.text_table = wandb.Table(columns=["epoch", "text"])

    def do_gen(self, model):
        df_gen = get_model_generations(model, model.tokenizer, max_new_tokens=64, N=1)
        s = display_gen(df_gen, with_q=False)
        return df_gen, s
        # can log?

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if n % self.every == 0:
            logger.info(f"\nGenerated on batch {batch_idx}")
            df_gen, s = self.do_gen(trainer.model._model)
            for _logger in trainer.loggers:
                # logger.log_text(s, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
                _logger.log_metrics({'gen': s}, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def on_train_epoch_end(self, trainer, pl_module):
        logger.info(f"\nGenerated at end of epoch {trainer.current_epoch}")
        df_gen, s = self.do_gen(trainer.model._model)
        for _logger in trainer.loggers:
            # logger.log_text(s, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
            _logger.log_metrics({'gen': s}, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

        if wandb.run:
            self.text_table.add_data(epoch, s)
