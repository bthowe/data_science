import sys
import joblib
import pandas as pd
from fuzzywuzzy import process

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def fuzzy_merge(df1, df2, how, on, thresh):
    lst = []

    df_merged = df1.merge(df2, how=how, on=on, indicator=True)
    for val in df_merged.query('_merge != "both"')[on].values:
        fuzzy_val = process.extract(val, df2[on].values.tolist(), limit=1)
        row = [val] + [fuzzy_val[0][0]] + [fuzzy_val[0][1]]
        lst.append(row)
        print(row)
    df_map = pd.DataFrame(
        lst,
        columns=[
            on,
            'closest_value',
            'closest_value_score'
        ]
    )

    df_fuzzy = df1.\
        merge(df_map, how='left', on=on, indicator=True).\
        query('_merge == "both"')

    df_fuzzy_merged = df_fuzzy.\
        query('closest_value_score >= {}'.format(thresh)).\
        drop([on, '_merge'], 1).\
        rename(columns={'closest_value': on, 'closest_value_score': '_match_score'}).\
        merge(df2, how=how, on=on).\
        reset_index(drop=True)

    df_fuzzy_unmerged = df_fuzzy.\
        query('closest_value_score < {}'.format(thresh)).\
        drop(['_merge', 'closest_value'], 1).\
        rename(columns={'closest_value_score': '_match_score'}).\
        merge(df2, how=how, on=on).\
        reset_index(drop=True)
    df_fuzzy_unmerged.loc[:, '_match_score'] = 'unmatched'

    df = df_merged.query('_merge == "both"').drop('_merge', 1)
    df['_match_score'] = 'match'
    return df.append(df_fuzzy_merged).append(df_fuzzy_unmerged).reset_index(drop=True)


def fuzzy_matches_n_lst(df1, df2, how, on, match_num):
    lst = []
    df_merged = df1.merge(df2, how=how, on=on, indicator=True)
    for val in df_merged.query('_merge != "both"')[on].values:
        fuzzy_val = process.extract(val, df2[on].values.tolist(), limit=match_num)
        row = [val]
        col_names = ['variable_value']
        for i in range(match_num):
            row += [fuzzy_val[i][0]] + [fuzzy_val[i][1]]
            col_names += ['candidate_match{}'.format(i), 'candidate_match{}_score'.format(i)]
        lst.append(row)
        print(row)
    return pd.DataFrame(
        lst,
        columns=col_names
    )

def main():
    df_b = pd.read_csv('df_b.csv')
    df_c = pd.read_csv('df_c.csv')

    df_b['name'] = df_b.apply(lambda x: x['v_b0'].lower() + ' ' + x['v_b1'].lower(), axis=1)
    df_c['name'] = df_c['v_c0'].str.lower()

    print(fuzzy_merge(df_b, df_c, 'left', 'name', 90))
    print(fuzzy_matches_n_lst(df_b, df_c, 'left', 'name', 3))

if __name__ == '__main__':
    main()

