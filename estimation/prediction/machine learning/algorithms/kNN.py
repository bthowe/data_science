
import numpy as np
# import bottleneck

# bottleneck.partition()



def data_create():
    return np.c_[np.random.uniform(-1, 1, size=[100, 2]), np.random.choice([0, 1], size=[100, 1])]

def feature_change(df):
    return df.assign(state=lambda x: x['fips'].map(state_name))

def data_out(df):
    df.to_csv('file path')

if __name__ == '__main__':
    data_create().\
        pipe(feature_change).\
        pipe(data_out)

    df = data_create()
    df_new = feature_change(df)
    feature_change(df_new)




#
#     # print(np.random.uniform(-1, 1, size=[2, 100]))
#     # print(np.random.choice([0, 1], size=[1, 100]))
#
#
# if __name__ == '__main__':
