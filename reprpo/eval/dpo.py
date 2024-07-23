from datasets import Dataset
from trl import DPOTrainer
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reprpo.helpers.adapters import set_adapter
from reprpo.helpers.torch import clear_mem

from datasets import load_dataset
import numpy as np


@torch.no_grad()
def eval_dpo_dataset(trainer: DPOTrainer, model: AutoModelForCausalLM, dataset: Dataset):
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    clear_mem(trainer)
    model.eval()
    model.config.use_cache = False


    data = []
    # use hf dpo trainer to tokenizer, and make loader
    dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=10)
    # eval_dataloader = trainer.get_eval_dataloader(dataset)
    eval_dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=trainer.args.eval_batch_size, collate_fn=trainer.data_collator)

    # model = trainer._wrap_model(trainer.model, training=False, dataloader=eval_dataloader)
    # model = trainer.accelerator.prepare_model(model, evaluation_mode=True)

    assert trainer.loss_type == 'ipo', 'only ipo is supported, since it gives us the avg of logps, and is not biased by response length'
    
    with torch.cuda.amp.autocast():
        for step, batch in enumerate(tqdm(eval_dataloader, unit='batch')):
            # batch = trainer._prepare_inputs(batch)

            forward_output = trainer.concatenated_forward(model, batch)
            (
                chosen_logps,
                rejected_logps,
            ) = forward_output[:2]

            # Note: if we are using ipo or reprpo this will be adjusted for length, but otherwise not which would bias the results
            logratio = chosen_logps-rejected_logps

            batch['chosen_input_ids'].shape
            batch['rejected_input_ids'].shape
            bs = batch['chosen_input_ids'].shape[0]
            i = bs * step + torch.arange(bs)
            data.append(dict(
                logratio=logratio.detach().cpu(),
                l_chosen=(batch['chosen_labels']>0).sum(-1).detach().cpu(),
                l_rejected=(batch['rejected_labels']>0).sum(-1).detach().cpu(),
                i=i
            ))
    # now concat the elements of data
    data = {k:torch.cat([d[k] for d in data], dim=0).numpy() for k in data[0].keys()}

    df = pd.DataFrame(data)
    df['correct'] = df['logratio'] > 0.

    # prob mass on correct answer
    odds = np.exp(df['logratio'])
    df['prob'] = odds / (1 + odds)
    return df

def eval_dpo_dataset_adapters(trainer, model, dataset, adapter_names = None, **kwargs):
    dfs = []
    if adapter_names is None:
        adapter_names = [None]+list(model.peft_config.keys())
    for adapter_name in tqdm(adapter_names, desc='adapters'):
        with set_adapter(model, adapter_name):
            print('adapter', adapter_name)
            df = eval_dpo_dataset(trainer, model, dataset, **kwargs)
        df['adapter'] = adapter_name if adapter_name is not None else 'base'
        dfs.append(df)
    
    df = pd.concat(dfs)
    return df


def ds_select(ds, N=np.inf):
    N = min(N, len(ds))
    return ds.select(range(N))


def load_tqa_dpo(N=None):

    slice = '' if N is None else f'[:{N}]' # https://huggingface.co/docs/datasets/en/loading#slice-splits
    dataset_tqab = load_dataset("EleutherAI/truthful_qa_binary", split=f'validation{slice}')
    def proc(row):
        l = row['label']
        return {
            'prompt': row['question'],
            'chosen': row['choices'][l],
            'rejected': row['choices'][~l],
        }
    return dataset_tqab.map(proc)

def eval(trainer, model, N=1000):



    slice = f'[0:{N}]' if N is not None else ''
    


    # TODO some of the https://github.dev/AI-secure/DecodingTrust datasetsm e.g. turned into dpo. E.g. ethics

    dataset1 = load_dataset('Atsunori/HelpSteer2-DPO', split=f'validation{slice}').rename_column('chosen_response', 'chosen').rename_column('rejected_response', 'rejected') # training set
    print('ds1')
    dataset2 = load_tqa_dpo(N) # OOS honesty in the fase of ~800 common misconceptions\
    print('ds2')
    dataset3 = load_dataset('unalignment/toxic-dpo-v0.2', split=f'train{slice}')
    print('ds3')

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
    clear_mem()
    print('clearedmem')

    # crop of N

    dfs = []
    for dataset_name, dataset in tqdm(datasets.items(), total=len(datasets), desc='datasets'):
        print(dataset_name)
        df = eval_dpo_dataset_adapters(trainer, model, dataset)
        df['dataset'] = dataset_name
        dfs.append(df)
    df = pd.concat(dfs)

    res =  df.groupby(['adapter', 'dataset'], dropna=False).mean()['correct'].unstack()
    return res, df
