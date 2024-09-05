
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# from peft import prepare_model_for_kbit_training
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


# https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/alpaca.jinja
ALPACA_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] | trim + '\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ bos_token + system_message }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### Instruction:\n' + message['content'] | trim + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### Response:\n' + message['content'] | trim + eos_token + '\n\n' }}
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
    {{ '### Response:\n' }}
{% endif %}"""

# this is from trl https://github.com/huggingface/trl/blob/cbcaa46cd3c02c0e7f724b764c5848ae73796de7/trl/trainer/utils.py#L747
# not sure if it's needed but `prepare_model_for_kbit_training` doesn't seem to do this ,despite claiming to
def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
            print(f"casting {name} to bf16")

def load_model(model_name, load_in_4bit=True, load_in_8bit=False, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16):
    model = None
    clear_mem()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        print(f"Not default chat template for {model_name}. Setting to ALPACA_CHAT_TEMPLATE")
        tokenizer.chat_template = ALPACA_CHAT_TEMPLATE

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            # https://huggingface.co/docs/transformers/v4.43.2/quantization/bitsandbytes#4-bit-qlora-algorithm
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype, # faster
            bnb_4bit_quant_type="nf4", # for training  normalized float 4 bit data type
            # skip_modules=[], # by default the head is kept in original precision
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                #  low_cpu_mem_usage=True,
                                                 quantization_config = quantization_config,
        torch_dtype=torch_dtype, 
        attn_implementation=attn_implementation,
        )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False    
    model.tokenizer = tokenizer

    # peft_module_casting_to_bf16(model)
    # model = prepare_model_for_kbit_training(model, {'use_gradient_checkpointing': args.use_gradient_checkpointing})

    return model, tokenizer


