from optuna.visualization._param_importances import _get_importances_infos
from optuna.importance import PedAnovaImportanceEvaluator
import pandas as pd


def optuna_df(study, key_metric):
   o, = _get_importances_infos(study, 
                        evaluator=PedAnovaImportanceEvaluator(), 
                        params=None, 
                     target=None,
                     target_name=key_metric,
                        )
   s_imp = pd.Series(o.importance_values, index=o.param_names)
   s_b = pd.Series(study.best_trial.params)
   df = study.trials_dataframe().query('state == "COMPLETE"')
   n = len(df)
   df = pd.concat([s_imp, s_b], axis=1, keys=['importance', f'best[N={n}]']).sort_values('importance', ascending=False)
   return df
