from datasets import load_dataset
from open_pref_eval.evaluation import evaluate, evaluate_model
import pandas as pd
from open_pref_eval.datasets import get_default_datasets
from reprpo.helpers.adapters import set_adapter

def evaluate_adapters(model, tokenizer, N=30, **kwargs):
    """
    use the open_pref_eval library

    to eval the model and it's adapters
    """


    dataset_helpsteer2 = load_dataset('Atsunori/HelpSteer2-DPO', split=f'validation[:{N}]', keep_in_memory=False).rename_column('chosen_response', 'chosen').rename_column('rejected_response', 'rejected') # training set

    datasets = get_default_datasets(N) + [dataset_helpsteer2]
    model.eval()

    return evaluate_model(datasets=datasets, model=model, tokenizer=tokenizer, **kwargs)

    # dfs = []
    # for adapter in adapters:
    #     with set_adapter(model, adapter):
    #         print(f'Adapter: {adapter}: {model.active_adapter}')
    #         _, df_res2 = evaluate(datasets, model=model, tokenizer=tokenizer, **kwargs)
    #     df_res2['adapter'] = adapter if adapter is not None else 'base'
    #     dfs.append(df_res2)
    # df_res = pd.concat(dfs)

    # df_agg =  df_res.groupby(['dataset', 'adapter'], dropna=False)['prob'].mean().unstack()
    # return df_agg, df_res
