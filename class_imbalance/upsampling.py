import sys
import numpy as np
import pandas as pd


def resample(y):
    class_count_dict = {y_val: y.loc[y == y_val].shape[0] for y_val in y.unique()}
    max_class_class = max(class_count_dict, key=class_count_dict.get)
    max_class_count = class_count_dict[max_class_class]

    sample_index_dict = {}
    for k, v in class_count_dict.iteritems():
        if k != max_class_class:
            sample_size = max_class_count - v
            sample_index_dict[k] = np.random.choice(y.loc[y == k].index, size=sample_size, replace=True)

    return sample_index_dict


def upsample(X, y):
    """Balances classes by resampling with replacement."""

    for k, v in resample(y).iteritems():
        X = X.append(X.iloc[v])
        y = y.append(y.iloc[v])

    return X.reset_index(drop=True), y.reset_index(drop=True)


if __name__ == '__main__':
    X = pd.DataFrame(np.random.uniform(-1, 1, size=(20, 5)), columns=['one', 'two', 'three', 'four', 'five'])

    np.random.seed(22)
    y = pd.Series(np.random.randint(0, 4, size=(20,)))
    print upsample(X, y)
