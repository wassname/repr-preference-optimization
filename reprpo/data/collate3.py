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
from open_pref_eval.trainer import apply_chat_template_to_completion, FALCAN_CHAT_TEMPLATE
from loguru import logger


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


# TODO replace with the open_pref_eval version. I would need to change this, and also use their collator, and then change forward to take 3 seperate inputs, so use concatednate input
@dataclass
class TokenizeRow:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    max_prompt_length: int = 128

    def __call__(self, batch: LazyRow) -> Dict[str, Any]:
        out = {}

        if self.tokenizer.chat_template is None:
            logger.warning(
                "No chat template set for tokenizer, using default FALCAN INSTRUCTR template."
            )
            self.tokenizer.chat_template = FALCAN_CHAT_TEMPLATE
        
        # TODO handle both being messages
        batch['prompt'], batch['chosen'] = apply_chat_template_to_completion(self.tokenizer, batch["prompt"], batch["chosen"])
        batch['rejected'] = apply_chat_template_to_completion(self.tokenizer, batch["prompt"], batch["rejected"])[1]

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
            self.max_length - len(prompt)
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

                encoded_inputs = {
                    "input_ids": ids,
                    "special_tokens_mask": self.tokenizer.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    ),
                }

                # FIXME do I want to pad here or later? replace with open_pref_eval version
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
        out["prompt_mask"] = [1] * len(prompt) + [0] * (self.max_length - len(prompt))
        return out
