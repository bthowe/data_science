import pandas as pd

def correlation_of_predicted_scores_categorical(df, group_var, covar):
    """Finds the Spearman correlation of the ranks for the mean scores for each value of 'covar' across the groups in
    'group.'"""

    # find common values of covar across all groups
    covar_set = set(df[covar].values)
    for group in df[group_var].unique():
        covar_set = covar_set.intersection(df.loc[df[group_var] == group, covar].values)

    # discard observations with covar values different than in covar_set
    df = df.\
        dropna(subset=[covar], axis=0).\
        loc[df[covar].isin(covar_set)]

    # create data set
    column_names = ['prediction_proba_rank_{}'.format(group) for group in df[group_var].unique()]
    df_covar_by_group = pd.DataFrame(index=list(covar_set), columns=column_names)

    # fill in columns
    for group in df[group_var].unique():
        df_temp = df.loc[df[group_var] == group]
        df_covar_by_group['prediction_proba_rank_{}'.format(group)] = df_temp['prediction_proba'].groupby(df_temp[covar]).mean().rank(ascending=False)

    print(df_covar_by_group.corr(method='spearman'))
