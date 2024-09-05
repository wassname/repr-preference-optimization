"""
see 
- https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L349
- https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L98
- https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
"""
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

# TODO this is all more complicated so we can do it per batch, but that doesn't really help with gpu mem, which is determined by max sequence length. We should just simplify this to be per dataset

def tokenize_row(feature: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """
    Tokenize a single row from a DPO specific dataset.

    usage:
        dataset2.map(lambda x: tokenize_row(x, tokenizer), batched=True, writer_batch_size=10)

    see https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L784
    """
    batch = {}
    # don't truncate or add special tokens yet
    batch["chosen"] = tokenizer(feature["chosen"], add_special_tokens=False)["input_ids"]
    batch["rejected"] = tokenizer(feature["rejected"], add_special_tokens=False)["input_ids"]
    batch["prompt"] = tokenizer(feature["prompt"], add_special_tokens=False)["input_ids"]
    return batch

@dataclass
class DPODataCollatorWithPadding:
    tokenizer: Any
    max_length: int
    pad_token_id: int
    mask_prompt_tokens:bool=True
    max_prompt_length: int=64
    device: str = "cpu"
    truncate_prompt_left: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """use to pad the sequences in each batch to an equal length so that we can assemble them in batches

        takes in batch of tokenized
        """
        # Initialize lists to hold batch data
        batch_data = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "rejected_mask": [],
            "chosen_mask": []

        }

        # Determine the longest sequence to set a common padding length
        max_length_common = 0
        if batch:
            for key in ["chosen", "rejected"]:
                current_max = max(len(item[key])+1 for item in batch) + self.max_prompt_length
                max_length_common = max(max_length_common, current_max)


        # Process each item in the batch
        for item in batch:
            prompt = item["prompt"]
            batch_data["prompt"].append(torch.tensor(item["prompt"]))

            # Do we want to crop the beginning or end of the prompt?
            if self.truncate_prompt_left:
                prompt = prompt[-self.max_prompt_length:]
            else:
                prompt = prompt[:self.max_prompt_length]

            for key in ["chosen", "rejected"]:
                # Adjust padding according to the common maximum length
                sequence = prompt + item[key]
                padded = sequence + [self.pad_token_id] * (max_length_common - len(sequence))
                mask = torch.ones(len(padded)).bool()

                # Set mask for all padding tokens to False
                mask[len(sequence):] = False

                batch_data[key].append(torch.tensor(padded))
                batch_data[f"{key}_mask"].append(mask)

        # Final processing
        for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
            # Stack all sequences into a tensor for the given key
            tensor_stack = torch.stack(batch_data[key])

            # Optionally truncate to maximum sequence length
            if self.max_length is not None:
                tensor_stack = tensor_stack[:, :self.max_length]

            # Move to the specified device
            batch_data[key] = tensor_stack.to(self.device)

        return batch_data
