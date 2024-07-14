# https://github.dev/sylinrl/TruthfulQA/blob/fdd8ad1c0d00a478cf8b0bb41a3ad8378c16293b/truthfulqa/models.py#L311
# - in 
# https://github.com/sylinrl/TruthfulQA
# FIXME there's something wrong here, scores too low?

from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from collections import defaultdict
from reprpo.helpers.adapters import set_adapter

from jaxtyping import Float, Int
from typing import Tuple
from torch import Tensor

def sum_select_choices_from_logits(log_probs: Float[Tensor, 'b h'], choice_ids: Int[Tensor, 'b c n']) -> Float[Tensor, 'b c n']:
    """sum the logits for each set of choices"""
    device = log_probs.device
    flat_choice_ids = rearrange(choice_ids, 'b c n -> b (c n)').to(device) # flatten
    flat_choice_logps = torch.gather(log_probs, 1, flat_choice_ids.long())
    choice_logps = rearrange(flat_choice_logps, 'b (c n) -> b c n', c=choice_ids.shape[1]) # unflatten
    return choice_logps

def calc_mc_log_ratio(last_token_logits: Float[Tensor, 'b h'], choice_ids: Int[Tensor, 'b c n'], labels: Int[Tensor, 'b ...']) -> Tuple[Float[Tensor, 'b c'], Float[Tensor, 'b']]:
    """multichoice log ratio."""
    logp = last_token_logits.log_softmax(-1)
    per_token_logps = sum_select_choices_from_logits(
            logp, choice_ids
        )
    # per_token_logps = per_token_logps.exp().sum(-1).log() # combine categories of tokens
    per_token_logps = torch.logsumexp(per_token_logps, -1) # combine categories of tokens

    # select the answer
    logp_right = torch.gather(per_token_logps, 1, labels[:, None].long())
    logp_wrong = (per_token_logps.exp().sum()-logp_right.exp()).log()
    log_ratio = logp_right - logp_wrong
    return per_token_logps, log_ratio

@torch.no_grad
def eval_tqa(model, tokenizer, dataset2, choice_ids, adapter_names = None):
    model.config.use_cache = False
    if adapter_names is None:
        adapter_names = [None]+list(model.peft_config.keys())
    data = defaultdict(list)
    model.eval()

    batch_size = 5
    dl = DataLoader(dataset2, batch_size=batch_size, num_workers=0)
    for i, b in enumerate(tqdm(dl)):
        inputs = {
            "input_ids": b["input_ids"].to(model.device),
            "attention_mask": b["attention_mask"].to(model.device),
        }
        for adapter_name in adapter_names:
            with torch.cuda.amp.autocast():
                if adapter_name is not None:
                    model.set_adapter(adapter_name)
                with set_adapter(model, adapter_name):
                    out = model(**inputs, use_cache=False)

            for j in range(len(out["logits"])):
                n = b["num_choices"][j]
                b_choice_ids = torch.tensor(choice_ids[:n]).unsqueeze(0).to(model.device).unsqueeze(-1)
                label = b["label"][j, 0]

                per_token_logps, log_ratios = calc_mc_log_ratio(out["logits"][j, -1][None], b_choice_ids, label[None].cuda())
                
                ans = tokenizer.batch_decode(out["logits"][j, -1][None].argmax(-1))[0]

                data[adapter_name or 'None'].append(dict(
                    ratios=log_ratios.exp().item(),
                    coverage=per_token_logps.exp().sum().item(),
                    ans=ans,
                    i=i*batch_size+j,
                ))
    dfs = []
    for k in data.keys():
        df = pd.DataFrame(data[k])
        df['adapter'] = k
        dfs.append(df)

    df = pd.concat(dfs)
    df['%'] = (df['ratios']/(df['ratios']+1)) * 100
    return df
