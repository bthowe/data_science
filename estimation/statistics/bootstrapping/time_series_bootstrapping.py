import sys
import joblib
import numpy as np
import pandas as pd
from patton_block_selection_algorithm import opt_block_length
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap
# stationary gives us a a stationary time-series, but tends to be slightly slower to converge.

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# todo: does this do overlapping blocks?
def block_bootstrap(df):
    bstar = opt_block_length(df[['target']], bootstrap_type='Circular', rnd=True)
    bs = CircularBlockBootstrap(bstar, df)
    for data in bs.bootstrap(100):
        print data[0][0]
        sys.exit()

# what is the optimal k? http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.558.1087&rep=rep1&type=pdf

if __name__ == '__main__':
    data = joblib.load('../sample_data_files/X_train.pkl')  # this data isn't time series data
    data['target'] = np.random.uniform(.5, 1.5, size=(len(data), 1))
    block_bootstrap(data)

