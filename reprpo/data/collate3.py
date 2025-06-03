"""
A simpler DPO tokenizer

Instead of doing a batch we just do all rows to max length
and we reuse tokenizer functions more
see 
- https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L349
- https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L98
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

"""
from contextlib import contextmanager
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict
from datasets.formatting.formatting import LazyRow
from dataclasses import dataclass


@contextmanager
def tok_setings(tokenizer: PreTrainedTokenizerBase, **kwargs):
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
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    max_prompt_length: int = 128

    def __call__(self, batch: LazyRow) -> Dict[str, Any]:
        out = {}

        # encode and truncate prompt
        with tok_setings(self.tokenizer, truncation_side="left"):
            prompt = self.tokenizer.encode_plus(
                batch["prompt"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )["input_ids"]
        max_ans_length = (
            self.max_length - len(prompt) - self.tokenizer.num_special_tokens_to_add()
        )

        with tok_setings(self.tokenizer, padding_side="right", truncation_side="right"):
            for key in ["chosen", "rejected"]:
                encoded_inputs = {}
                # tokenizer and truncate
                ans = self.tokenizer.encode_plus(
                    batch[key],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_ans_length,
                )["input_ids"]
                out[key + "_truncated"] = len(ans) >= max_ans_length

                ids = prompt + ans

                # add special tokens
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                encoded_inputs = {
                    "input_ids": ids,
                    "special_tokens_mask": self.tokenizer.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    ),
                }

                # pad and attention mask
                encoded_inputs = self.tokenizer.pad(
                    encoded_inputs,
                    max_length=self.max_length,
                    padding="max_length",
                    return_attention_mask=True,
                )

                out[key] = encoded_inputs["input_ids"]
                out[key + "_mask"] = encoded_inputs["attention_mask"]
                out[key + "_special_tokens_mask"] = encoded_inputs["special_tokens_mask"]

        out["prompt_truncated"] = len(prompt) >= self.max_prompt_length

        # we want to update the prompt with the special tokens
        right_special_tokens = out[key + "_special_tokens_mask"][len(prompt):].sum()
        prompt = self.tokenizer.build_inputs_with_special_tokens(prompt)[:-right_special_tokens]
        out["prompt_mask"] = [1] * len(prompt) + [0] * (self.max_length - len(prompt))
        return out
