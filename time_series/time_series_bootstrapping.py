import sys
import numpy as np
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap



# todo: (1) block bootstrapping, (2) re-sampling y's, (3) re-sampling the errors
# anything that involves sampling the X's won't work because of intertemporal dependence. I would have to use block
# bootstrapping
# however, taking the x's and (1) drawing from the conditional distribution of y, or (2) sampling with replacement from
# the errors, and then backing out the y would work.



def outcome_condition_distribution_bootstrap(mode, X, y):
    pass


def error_resampling_bootstrap(model, X, y):
    e = y - model.predict(X)
    e_resampled = np.random.choice(e, size=len(y), replace=True)
    return model.predict(X) + e_resampled

def block_bootstrap():
    y = np.random.standard_normal((6, 1))
    print y
    x = np.random.standard_normal((6, 2))
    print x
    z = np.random.standard_normal(6)
    print z
    bs = CircularBlockBootstrap(2, x, y=y, z=z)
    for data in bs.bootstrap(100):
        print data; sys.exit()
        bs_x = data[0][0]
        bs_y = data[1]['y']
        bs_z = bs.z

# I only need to pass x in.
# todo: I have my data to pass in, but what should the block size be?

if __name__ == '__main__':
    model = None
    X = None
    y = None
    error_resampling_bootstrap(model, X, y)