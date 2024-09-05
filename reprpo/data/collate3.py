"""
A simpler DPO tokenizer

Instead of doing a batch we just do all rows to max length
and we reuse tokenizer functions more

"""
from contextlib import contextmanager
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List
from dataclasses import dataclass
import torch

@contextmanager
def tok_setings(tokenizer: PreTrainedTokenizer, **kwargs):
    """
    Temporarily set the tokenizer settings to the values in kwargs
    """
    old_settings = {k: getattr(tokenizer, k) for k in kwargs.keys()}
    try:
        for k, v in kwargs.items():
            setattr(tokenizer, k, v)
        yield tokenizer
    finally:
        for k, v in old_settings.items():
            setattr(tokenizer, k, v)

@dataclass
class TokenizeRow:
    tokenizer: PreTrainedTokenizer
    max_length: int
    max_prompt_length: int=64

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        out = {}

        # encode and truncate prompt
        with tok_setings(self.tokenizer, truncation_side='left'):
            prompt = self.tokenizer.encode_plus(batch['prompt'], add_special_tokens=False, padding=False, truncation=True, max_length=self.max_prompt_length)['input_ids']
        max_ans_length=self.max_length - len(prompt) - self.tokenizer.num_special_tokens_to_add()

        with tok_setings(self.tokenizer, truncation_side='right'):
            for key in ["chosen", "rejected"]:
                encoded_inputs = {}
                # tokenizer and truncate
                ans = self.tokenizer.encode_plus(batch[key], add_special_tokens=False, truncation=True, max_length=max_ans_length)['input_ids']            
                out[key+'_truncated'] = len(ans)>=max_ans_length
                
                ids = prompt + ans

                # add special tokens
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                encoded_inputs = {
                    "input_ids": ids,
                    "special_tokens_mask": self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                }

                # pad and attention mask
                encoded_inputs = self.tokenizer.pad(encoded_inputs, max_length=self.max_length, padding='max_length', return_attention_mask=True)

                out[key] = encoded_inputs['input_ids']
                out[key+'_mask'] = encoded_inputs['attention_mask']

        # I also want to output a token mask but it's complicated by the special tokens
        # FIXME: this assumes one bos and on eos token
        out['prompt'] = self.tokenizer.build_inputs_with_special_tokens(prompt)[:-1]
        out['prompt_mask'] = [1] * len(out['prompt']) + [0] * (self.max_length - len(out['prompt']))
        out['prompt_truncated'] = len(out['prompt'])>=self.max_prompt_length

        return out


