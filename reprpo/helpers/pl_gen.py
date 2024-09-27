from reprpo.gen import get_model_generations, display_gen
from lightning.pytorch.callbacks import Callback


class GenCallback(Callback):
    """A callback to generate on sample each N samples."""

    def __init__(self, every=50):
        self.every = every

    def do_gen(self, model):
        df_gen = get_model_generations(model, model.tokenizer, max_new_tokens=42, N=1)
        display_gen(df_gen, with_q=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if n % self.every == 0:
            print(f"\nGenerated on batch {batch_idx}")
            self.do_gen(trainer.model._model)

    def on_train_epoch_end(self, trainer, pl_module):
        self.do_gen(trainer.model._model)
