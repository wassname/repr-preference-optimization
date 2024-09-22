import gc, torch


def clear_mem(trainer=None):
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

    if trainer is not None:
        trainer.accelerator.free_memory()
    torch.cuda.empty_cache()

    gc.collect()
