import os
os.environ["WANDB_GROUP"] = "sweep4"

# for H100
batch_sizes = {
    "wassname/Qwen3-0.6B-sft": 18,
    "allenai/OLMo-2-0425-1B-SFT": 13, # error please specifcy target modules in config
    # "wassname/llama-3-2-1b-sft",
    "wassname/llama-3.2-3b-sft": 10,
    "princeton-nlp/Llama-3-Base-8B-SFT": 5,
    # "allenai/Llama-3.1-Tulu-3-8B-SFT": 5,
    # add large models too if you want custom sizes
}
default_batch_size = 7

adapters = [
    "hs-none-InnerDPO",
    "dpo",
    "hs-supr-InnerDPO",
    "hs-ether-InnerDPO",
    "side-none-InnerDPO",
    # "projgrad",
]

seeds = [
    1, 2, 3
]

datasets = [
    # set 1
    "math",
    "alpaca_mmlu",
    "code_easy",
    "cooking",

    "alpaca_low_quality",
    # "math_easy",

    # # set 2
    # "code_easy",
    # "us_history",
    # "change_my_view",
    # "raven_matrices",

    # # set 3
    # "ranking_logic_easy",
    # "shp_low_quality",
    # "pursue_goals",
    # "creative_writing",
    # "alpaca_easy",
    # "arc_easy",
    # "us_history_textbook",
    # "alpaca_chat",
    # "raven_easy",
    # "code_low_quality",
    # "alpaca_short",
]

from jinja2 import Environment, FileSystemLoader
import os
env = Environment(loader=FileSystemLoader('scripts'), trim_blocks=True, lstrip_blocks=True)
template = env.get_template('sweep.sh.j2')
rendered = template.render(
    batch_sizes=batch_sizes,
    default_bs=default_batch_size,
    adapters=adapters,
    seeds=seeds,
    datasets=datasets,
)
print(rendered)
