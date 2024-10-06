from optuna.visualization._param_importances import _get_importances_infos
from optuna.importance import PedAnovaImportanceEvaluator, FanovaImportanceEvaluator
import pandas as pd
from optuna import Study
import numpy as np


def optuna_df(study: Study, key_metric: str):
   n = len(study.trials)

   try:
      # note should be more than 50 studies
      # fanova 2014
      # ped-anove is 2024 and faster, I asume the older one is better 
      o, = _get_importances_infos(study, 
                           evaluator=FanovaImportanceEvaluator(), 
                           params=None, 
                        target=None,
                        target_name=key_metric,
                           )
      s_imp = pd.Series(o.importance_values, index=o.param_names)
   except ValueError:
       df = pd.DataFrame(columns=['importance', 'best'])
       df.index.name = f'{study.study_name} N={n}'
       return df
   
   try:
      # Study instance does not contain completed trials.
      s_b = pd.Series(study.best_trial.params)
      v = study.best_trial.values[0]
   except ValueError:
      s_b = pd.Series([])
      v = np.nan
   
   n = len(study.trials)
   if n==0:
      n_c = 0
   else:
      df = study.trials_dataframe()
      n_c = len(df.query('state == "COMPLETE"'))

   df = pd.concat([s_imp, s_b], axis=1, keys=['importance', f'best']).sort_values('importance', ascending=False)
   # round importance to 3 decimal places
   df['importance'] = df['importance'].round(3)
   df.index.name = f'{study.study_name} N=✓{n_c}/{n}, best={v:.3f}'

   # df2 = df.copy()
   # df2 = df2.style.background_gradient(cmap='viridis', axis=0)
   # df2.set_caption(study.study_name)
   return df


from optuna.importance._base import _get_distributions
def get_params(study):
   r = _get_distributions(study, None)
