


sft:
    # On 4 80GB A100s, SFT training took about 1hrs.
    ./.venv/bin/python -u train.py \
        model=blank_model model.name_or_path=NousResearch/Meta-Llama-3-8B \
        model.block_name=LlamaDecoderLayer \
        datasets=[hh,shp] \
        loss=sft \
        exp_name=anthropic_shp_sft_llama_7b \
        gradient_accumulation_steps=2 \
        batch_size=64 \
        eval_batch_size=32 \
        trainer=FSDPTrainer \
        sample_during_eval=false

dpo:
    # On 4 80GB A100s, DPO training took about 2hrs 45min.
    ./.venv/bin/python -u train.py \
        model=pythia69 \
        datasets=[hh] \
        loss=dpo \
        loss.beta=0.1 \
        model.archive=/path/to/checkpoint/from/sft/step-XXXX/policy.pt exp_name=anthropic_dpo_pythia69 \
        gradient_accumulation_steps=2 \
        batch_size=32 \
        eval_batch_size=32 \
        trainer=FSDPTrainer sample_during_eval=false
