
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from reprpo.helpers.torch import clear_mem


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(model_name, bnb=True):
    model = None
    clear_mem()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if bnb:
        quantization_config = BitsAndBytesConfig(
            # https://huggingface.co/docs/transformers/v4.43.2/quantization/bitsandbytes#4-bit-qlora-algorithm
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # faster
            bnb_4bit_quant_type="nf4", # for trainng
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            quantization_config = quantization_config,
            # load_in_4bit=True,
            # attn_implementation="flash_attention_2",
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_quant_type="nf4",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False    
    model.tokenizer = tokenizer

    return model, tokenizer


