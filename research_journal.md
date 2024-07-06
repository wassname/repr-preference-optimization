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
