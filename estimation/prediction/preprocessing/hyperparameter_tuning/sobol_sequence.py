import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

# SALib documentation: https://salib.readthedocs.io/en/latest/getting-started.html


def coverage():
    N = 50
    D = 2

    problem = {
        'num_vars': D,
        'names': ['x1', 'x2'],
        'bounds': [
            [-1, 1],
            [-1, 1]
        ]
    }
    sobol = saltelli.sample(problem, N, calc_second_order=False)
    random = np.random.uniform(-1, 1, size=(N * (2 * D + 2), 2))


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(sobol[:, 0], sobol[:, 1], color='cornflowerblue')
    plt.axis('scaled')

    ax = fig.add_subplot(2, 1, 2)
    ax.scatter(random[:, 0], random[:, 1], color='firebrick')
    plt.axis('scaled')
    plt.show()

    # The sobol sequence has a more even coverage, visually


def data_create():
    N = 1000
    error_scale = 10

    b0 = .5
    b1 = 4
    b2 = -7
    b3 = 20

    X = np.c_[np.ones(shape=(N, 1)), np.random.uniform(0, 10, size=(N, 3))]
    y = b0 + X[:, 0] * b1 + X[:, 1] * b2 + X[:, 2] * b3 + np.random.normal(0, error_scale)

    return X, y

def _hyper_params_create(second_order=True):
    problem = {
        'num_vars': 1,
        'names': ['alpha'],
        'bounds': [[0, 2]]
    }

    if second_order:  # includes second-order indices
        return saltelli.sample(problem, 1000)  # N * (2D + 2); here, N = 10, D = 3

    # excludes second-order indices
    return saltelli.sample(problem, 1000, calc_second_order=False)  # N * (D + 2); here, N = 10, D = 3

def hyperparameter_tune(X, y):
    params = _hyper_params_create()

    best_params = {'parameters': [1], 'cross_val_score': -1e6}
    for row in params:
        cross_val = cross_validate(Ridge(alpha=row[0], fit_intercept=False), X, y, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=10)

        if np.mean(cross_val['test_score']) > best_params['cross_val_score']:
            best_params['parameters'] = row
            best_params['cross_val_score'] = np.mean(cross_val['test_score'])

    print('Best parameters: {}'.format(best_params))
    return best_params['parameters']

def model_train(X, y, params):
    r = Ridge(alpha=params[0], fit_intercept=False)
    r.fit(X, y)
    return r

def main():
    X, y = data_create()
    params = hyperparameter_tune(X, y)
    model = model_train(X, y, params)

    print(model.coef_)  # the first coefficient corresponds to the intercept


if __name__ == '__main__':
    coverage()
    # main()
