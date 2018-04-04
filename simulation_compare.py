import sys
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
def data_create():

    df = pd.DataFrame(np.random.uniform(size=(n, 1)), columns=['prob'])

    joint_probs = [.1, .5, .9, 1]
    for ind, val in enumerate(joint_probs):
        df[ind] = 0
        df.loc[df['prob'] < joint_probs[ind], ind] = 1
    df['sum'] = df[0] + df[1] + df[2] + df[3]

    def one_two(x):
        # print(x); sys.exit()
        if x['sum'] == 1:
            x['one'] = 1
            x['two'] = 1
        elif x['sum'] == 2:
            x['one'] = 1
            x['two'] = 0
        elif x['sum'] == 3:
            x['one'] = 0
            x['two'] = 1
        else:
            x['one'] = 0
            x['two'] = 0
        return x

    return df.apply(one_two, axis=1)[['one', 'two']]

def iterative_pull(df):
    n = 10000
    cols = ['one', 'two']
    df_new = []
    for i in range(n):
        ind = i % 2
        col1 = cols[ind]
        val1 = int(df[col1].sample(n=1))
        col2 = cols[1 - ind]

        val2 = int(df.query('{0} == {1}'.format(col1, val1))[col2].sample(n=1))

        df_new.append([val1, val2])
    return pd.DataFrame(df_new, columns=['one', 'two'])


if __name__ == '__main__':
    data = data_create()
    print(pd.crosstab(data['one'], data['two']).apply(lambda r: r / n, axis=1))
    data1 = iterative_pull(data)
    print(pd.crosstab(data1['one'], data1['two']).apply(lambda r: r / 10000, axis=1))



