# 2024-07-06 16:21:34



```sh
# run SFT
python -u train.py \
    model=blank_model model.name_or_path=/PATH/TO/LLAMA/WEIGHTS \
    model.block_name=LlamaDecoderLayer \
    datasets=[hh,shp] \
    loss=sft exp_name=anthropic_shp_sft_llama_7b \
    gradient_accumulation_steps=2 batch_size=64 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false

```



this fork adds drpo https://github.com/eric-mitchell/direct-preference-optimization/compare/main...haobozhang:direct-preference-optimization:main

ompute the DPO loss for Plackett-Luce model https://github.com/spacegoing/pldpo/commit/80d7f1c1e0042f0858461b338cc4f5de7040a635


LORA https://github.com/gzliyu/direct-preference-optimization/commit/43147a224387b4047f5921e95d77a751243e29b0

https://github.com/andersonbcdefg/dpo-lora
https://github.com/unslothai/unsloth
unsloth/llama-3-8b-bnb-4bit",  
