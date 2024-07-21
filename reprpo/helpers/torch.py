import gc, torch

def clear_mem(trainer=None):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if trainer is not None:
        trainer.accelerator.free_memory()

    gc.collect()
