import gc, torch
def clear_mem():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
