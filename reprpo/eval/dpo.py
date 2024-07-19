from datasets import Dataset
from trl import DPOTrainer
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reprpo.helpers.adapters import set_adapter
from reprpo.helpers.torch import clear_mem

@torch.no_grad()
def eval_dpo_dataset(trainer: DPOTrainer, model: AutoModelForCausalLM, dataset: Dataset):
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    model.eval()
    model.config.use_cache = False


    data = []
    # use hf dpo trainer to tokenizer, and make loader
    dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=10)
    # eval_dataloader = trainer.get_eval_dataloader(dataset)
    eval_dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=trainer.args.eval_batch_size, collate_fn=trainer.data_collator)

    # model = trainer._wrap_model(trainer.model, training=False, dataloader=eval_dataloader)
    # model = trainer.accelerator.prepare_model(model, evaluation_mode=True)
    
    with torch.cuda.amp.autocast():
        for step, batch in enumerate(tqdm(eval_dataloader, unit='batch')):
            # batch = trainer._prepare_inputs(batch)

            forward_output = trainer.concatenated_forward(model, batch)
            (
                chosen_logps,
                rejected_logps,
            ) = forward_output[:2]

            # FIXME: if we are using ipo or reprpo this will be adjusted for length, but otherwise not which would bias the results
            logratio = chosen_logps-rejected_logps
            # TODO add length to check it not always the longest
            data.append(logratio.detach().cpu())
    data = torch.cat(data, dim=0)

    df = pd.Series(data).to_frame('logratio')
    df['correct'] = df['logratio'] > 0.
    return df

def eval_dpo_dataset_adapters(trainer, model, dataset, adapter_names = None, **kwargs):
    dfs = []
    if adapter_names is None:
        adapter_names = [None]+list(model.peft_config.keys())
    for adapter_name in tqdm(adapter_names, desc='adapters'):
        clear_mem()
        trainer.accelerator.free_memory()
        with set_adapter(model, adapter_name):
            df = eval_dpo_dataset(trainer, model, dataset, **kwargs)
        df['adapter'] = adapter_name if adapter_name is not None else 'base'
        dfs.append(df)
    
    df = pd.concat(dfs)
    return df


from datasets import load_dataset
import numpy as np

def ds_select(ds, N=np.inf):
    N = min(N, len(ds))
    return ds.select(range(N))


def load_tqa_dpo():
    dataset_tqab = load_dataset("EleutherAI/truthful_qa_binary")['validation']
    def proc(row):
        l = row['label']
        return {
            'prompt': row['question'],
            'chosen': row['choices'][l],
            'rejected': row['choices'][~l],
        }
    return dataset_tqab.map(proc)

def eval(trainer, model, N=1000):

    dataset1 = load_dataset('Atsunori/HelpSteer2-DPO')['validation'].rename_column('chosen_response', 'chosen').rename_column('rejected_response', 'rejected') # training set
    dataset2 = load_tqa_dpo() # OOS honesty in the fase of ~800 common misconceptions
    dataset3 = load_dataset('unalignment/toxic-dpo-v0.2')['train']

    # TODO add these other ones
    # ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

    # ['conversation_hash', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted', 'state', 'country', 'hashed_ip', 'header']
    # nontoxic_dataset = load_dataset("justinphan3110/wildchat_over_refusal", split="nontoxic").select(range(500))
    # toxic_dataset = load_dataset("justinphan3110/wildchat_over_refusal", split="nontoxic").select(range(500))
    # dataset = load_dataset("justinphan3110/harmful_harmless_instructions") ??
    
    datasets = {
        'train_HelpSteer2': dataset1, 'OOD_trufullqa': dataset2, 'OOD_toxic': dataset3
    }
    datasets = {k: ds_select(v, N) for k,v in datasets.items()}

    # crop of N

    dfs = []
    for dataset_name, dataset in tqdm(datasets.items(), item='dataset', total=len(datasets)):
        df = eval_dpo_dataset_adapters(trainer, model, dataset)
        df['dataset'] = dataset_name
        dfs.append(df)
    df = pd.concat(dfs)

    res =  df.groupby(['adapter', 'dataset'], dropna=False).mean()
    return res, df
