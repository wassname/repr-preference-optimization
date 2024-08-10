
import pandas as pd
import matplotlib.pyplot as plt

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("epoch").last().ffill().bfill()
    return df_histe

def plot_hist(df_hist, allowlist=None, logy=False):
    """plot groups of suffixes together"""
    suffixes = list(set([c.split('/')[-1] for c in df_hist.columns if '/' in c]))
    for suffix in suffixes:
        if allowlist and suffix not in allowlist: continue
        df_hist[[c for c in df_hist.columns if c.endswith(suffix) and '/' in c]].plot(title=suffix, style='.', logy=logy)
        plt.title(suffix)   
        plt.show()
