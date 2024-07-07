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


Why does the trainer log every step?
self.training_bar.write(str(logs))
https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L626
https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/utils/notebook.py#L335

# 2024-07-07 13:10:56

It's kinda working! Now for a direct comparison
- without sft
- shall I use GENIE? or a tiny subet https://github.com/Joshuaclymer/GENIES/blob/22c8afb2551851fb3f2d1a2dcf70e7608908f6b1/src/api/compute_generalization_metrics.py#L11
  - Train the base model on each source distribution and then evaluate it on the target distribution.
