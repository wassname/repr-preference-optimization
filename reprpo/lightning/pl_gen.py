from reprpo.eval.gen import get_model_generations, display_gen
from lightning.pytorch.callbacks import Callback


class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=50):
        self.every = every

    def do_gen(self, model):
        df_gen = get_model_generations(model, model.tokenizer, max_new_tokens=64, N=1)
        s = display_gen(df_gen, with_q=False)
        return df_gen, s
        # can log?

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if n % self.every == 0:
            print(f"\nGenerated on batch {batch_idx}")
            df_gen, s = self.do_gen(trainer.model._model)
            for logger in trainer.loggers:
                logger.log_text(s, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def on_train_epoch_end(self, trainer, pl_module):
        df_gen, s = self.do_gen(trainer.model._model)
        for logger in trainer.loggers:
            logger.log_text(s, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
