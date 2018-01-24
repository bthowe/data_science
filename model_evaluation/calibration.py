import numpy as np
import pandas as pd

def calibrate_probas(model, X, y, bin_width=.1):
    bins = np.linspace(0, 1, (1 / bin_width) + 1)
    proba_table_columns = ['bins', 'bin_lower_bound', 'bin_upper_bound', 'count', 'probability']

    df = X.assign(predicted_probas=model.predict_proba(X)[:, 1], target=y)[['predicted_probas', 'target']]  # do I want to include any other feature in this dataset?
    df = df.assign(bins=pd.cut(df['predicted_probas'], bins=bins, include_lowest=True))
    df = df. \
        groupby(df['bins']). \
        agg({'predicted_probas': 'count', 'target': 'mean'}).\
        rename(columns={'predicted_probas': 'count', 'target': 'probability'}). \
        reset_index(). \
        assign(bin_lower_bound=np.linspace(0, 1 - bin_width, (1 / bin_width)), bin_upper_bound=np.linspace(bin_width, 1, (1 / bin_width))) \
        [proba_table_columns]

    # if the count in a bin is fewer than 20, make probability null for that row
    df.loc[df['count'] < 20, 'probability'] = np.nan

    # interpolate missing values for interior bins
    df['probability'] = df['probability'].interpolate(method='piecewise_polynomial')

    # extrapolate missing values for the first and last bins
    df_extrapolate = df.query('probability == probability')
    p = np.polyfit(df_extrapolate.index.tolist(), df_extrapolate['probability'].values, 2)
    func = lambda x: p[0] * x ** 2 + p[1] * x + p[2]
    def row_name(x):
        if x['probability'] != x['probability']:
            return func(x.name)
        else:
            return x['probability']
    df['probability'] = df.apply(row_name, axis=1)

    return df