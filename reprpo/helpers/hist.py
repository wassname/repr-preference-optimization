from matplotlib import pyplot as plt
from trl import DPOTrainer
import pandas as pd
import numpy as np

def plot_hist(trainer: DPOTrainer):

    current_args = trainer.args.to_dict()
    default_args = type(trainer.args)('').to_dict()
    blocklist = ['output_dir', 'per_device_eval_batch_size', 'run_name' 'remove_unused_columns']
    args_diff = {k: v for k, v in current_args.items() if v != default_args[k]}
    args_diff = {k: v for k, v in args_diff.items() if k not in blocklist}


    plt.style.use('ggplot')
    df_hist1 = pd.DataFrame(trainer.state.log_history)
    df_hist1 = df_hist1.groupby('step').mean() # mean over gradient accum?
    df_hist2 = df_hist1.dropna(axis=1, thresh=int(len(df_hist1)*0.9))

    N = len(df_hist2.columns)
    fig, axes = plt.subplots(int(np.ceil(N/2)), 2, figsize=(12, N), sharex=True)
    axes = axes.flatten()
    for i, c in enumerate(df_hist2.columns):
        x = df_hist2[c].dropna()
        x.plot(ax=axes[i], style='.', ms=7, title=c)
        x.ewm(span=30).mean().plot(ax=axes[i])
    plt.tight_layout()

    return df_hist1, args_diff
