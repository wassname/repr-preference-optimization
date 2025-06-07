from reprpo.eval.gen import get_model_generations, display_gen
from lightning.pytorch.callbacks import Callback
from loguru import logger
import wandb

class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=150):
        self.every = every
        self.text_table = wandb.Table(columns=["epoch", "text"], log_mode="INCREMENTAL")

    def do_gen(self, trainer):
        step=trainer.fit_loop.epoch_loop._batches_that_stepped
        # step=trainer.fit_loop.epoch_loop.batch_idx

        epoch=trainer.current_epoch
        model = trainer.model._model

        df_gen = get_model_generations(model, model.tokenizer, max_new_tokens=64, N=2)
        gen_s = display_gen(df_gen, with_q=False)
        for _logger in trainer.loggers:
            # if hasattr(_logger, 'log_text'):
            #     _logger.log_text(gen_s, step=step) # fail artifact nam is longer than 128 chars
            _logger.log_metrics({'gen': gen_s})#, step=step)

        if wandb.run:
            self.text_table.add_data(epoch, gen_s)
            wandb.log({"training_table": self.text_table})#/;;;/., step=step)
        return df_gen, gen_s
        # can log?

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        
        if n % self.every == 0:
            logger.info(f"\nGenerated on batch {batch_idx}")
            df_gen, s = self.do_gen(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        logger.info(f"\nGenerated at end of epoch {trainer.current_epoch}")
        step=trainer.fit_loop.epoch_loop._batches_that_stepped
        df_gen, gen_s = self.do_gen(trainer)
