from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# from peft import prepare_model_for_kbit_training
from reprpo.helpers.torch import clear_mem
from loguru import logger
from open_pref_eval.helpers.load_model import load_hf_or_peft_model


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
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



# this is from trl https://github.com/huggingface/trl/blob/cbcaa46cd3c02c0e7f724b764c5848ae73796de7/trl/trainer/utils.py#L747
# not sure if it's needed but `prepare_model_for_kbit_training` doesn't seem to do this ,despite claiming to
def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
            logger.debug(f"casting {name} to bf16")

# fix me just replace with the one from open_pref_eval
def load_model(
    model_name,
    load_in_4bit=True,
    load_in_8bit=False,
    attn_implementation=None,#"flash_attention_2",
    torch_dtype=torch.bfloat16,
):
    model = None
    clear_mem()

    model, tokenizer = load_hf_or_peft_model(
        model_name,
        model=model,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError("This model does not have a chat template set. It might be a base model, but this method works on SFT models, not base or instruction/RLHF models.")

    
    # model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.tokenizer = tokenizer

    # peft_module_casting_to_bf16(model)
    # model = prepare_model_for_kbit_training(model, {'use_gradient_checkpointing': args.use_gradient_checkpointing})

    return model, tokenizer
