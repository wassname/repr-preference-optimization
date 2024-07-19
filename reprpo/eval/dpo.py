from datasets import Dataset
from trl import DPOTrainer
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd
import torch

from reprpo.helpers.adapters import set_adapter
from reprpo.helpers.torch import clear_mem

@torch.no_grad()
def eval_dpo_dataset(trainer: DPOTrainer, model: AutoModelForCausalLM, dataset: Dataset):
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer
    """
    model.eval()
    model.config.use_cache = False


    data = []
    # use hf dpo trainer to tokenizer, and make loader
    dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=10)
    eval_dataloader = trainer.get_eval_dataloader(dataset)
    
    for step, batch in enumerate(tqdm(eval_dataloader, unit='batch')):

        forward_output = trainer.concatenated_forward(model, batch)
        (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
        ) = forward_output[:4]

        # FIXME: if we are using ipo or reprpo this will be adjusted for length, but otherwise not which would bias the results
        logratio = chosen_logps-rejected_logps
        data.append(logratio.detach().cpu())
    data = torch.cat(data, dim=0)

    df = pd.Series(data.exp()).to_frame('prob')
    df['correct'] = df['prob'] > 0.5
    return df

def eval_dpo_dataset_adapters(trainer, model, dataset, adapter_names = None, **kwargs):
    clear_mem()
    dfs = []
    if adapter_names is None:
        adapter_names = [None]+list(model.peft_config.keys())
    for adapter_name in adapter_names:
        with set_adapter(model, adapter_name):
            df = eval_dpo_dataset(trainer, model, dataset, **kwargs)
        df['adapter'] = adapter_name
        dfs.append(df)
    
    df = pd.concat(dfs)
    return df
