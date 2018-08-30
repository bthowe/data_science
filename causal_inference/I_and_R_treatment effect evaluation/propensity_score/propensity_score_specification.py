import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from itertools import combinations_with_replacement
from sklearn.linear_model import LogisticRegression

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def _feature_keep(df):
    covars = ['treatment_group', 'monthly_premium', 'agent_tenure', 'policy_wait', 'state_id', 'payment_time']
    return df[covars]


def _feature_dummy_create(df):
    df['state_id'] = df['state_id'].astype(str)
    return pd.get_dummies(df)


def preprocess(df):
    return df.pipe(_feature_keep).pipe(_feature_dummy_create)


class PropensityScoreSpecification(object):
    """The purpose of this class is to (1) algorithmically identify a specification useful for estimating the
    propensity score, and (2) estimate the propensity score for each observations."""
    def __init__(self, X, t, basic_covars, linear_thresh=1, quad_thresh=2.71):
        self.X = X
        self.t = t
        self.basic_covars = basic_covars
        self.linear_thresh = linear_thresh
        self.quad_thresh = quad_thresh

        self.X_update = self.X[basic_covars]
        self.covar_lst = None

        self._candidate_covars = []
        self._lr = LogisticRegression()
        self._l0 = self._likelihood_calc(self.X_update)

    def _likelihood_calc(self, X):
        self._lr.fit(X, self.t)
        t_pred = self._lr.predict_proba(X)
        return -log_loss(self.t, t_pred)  # log-loss is the negative log likelihood

    def _choose_covar(self, method):
        score = {'score': 0, 'covar': '', 'data': None}
        for covar in self._candidate_covars:
            if method == 'quad':
                X_new = self.X_update[covar.split('__X__')[0]] * self.X_update[covar.split('__X__')[1]]
            else:
                X_new = self.X[covar]
            X_temp = pd.concat(
                [
                    self.X_update,
                    X_new
                ], axis=1
            )
            l1 = self._likelihood_calc(X_temp)

            new_score = 2 * (l1 - self._l0)
            # new_score = self._l0 / l1
            if score['score'] < new_score:
                score['score'] = new_score
                score['covar'] = covar
                score['data'] = X_temp
        return score

    def _iterate(self, method, thresh):
        go = True
        while go:
            score = self._choose_covar(method)  # find the covar with the highest lr statistic
            if score['score'] > thresh:  # this score must be greater than the threshold
                self.X_update = score['data']  # update the baseline dataset
                self._l0 = self._likelihood_calc(self.X_update)  # update the baseline likelihood
                self._candidate_covars.remove(score['covar'])  # remove the covariate from list of candidates
            else:
                go = False
            if not self._candidate_covars:
                go = False

    def _additional_linear_terms(self):
        self._candidate_covars = [col for col in self.X if col not in self.basic_covars]
        self._iterate('linear', self.linear_thresh)

    def _quadratic_interaction_terms(self):
        self._candidate_covars = ['__X__'.join(pair) for pair in combinations_with_replacement(self.X_update.columns, 2)]
        self._iterate('quad', self.quad_thresh)

    def specification_create(self):
        self._additional_linear_terms()
        self._quadratic_interaction_terms()
        self.covar_lst = self.X_update.columns.tolist()
        return self

    def score(self):
        self._lr.fit(self.X_update, self.t)
        return self._lr.predict_proba(self.X_update)[:, 0]

if __name__ == '__main__':
    df = pd.read_csv('test_data.csv').pipe(preprocess)
    t = df.pop('treatment_group')

    basic_covariates = ['monthly_premium', 'policy_wait']
    pss = PropensityScoreSpecification(df, t, basic_covariates)
    print(pss.specification_create().score())

    ## How to use
    # df = pd.read_csv('test_data.csv').pipe(preprocess)
    # t = df.pop('treatment_group')
    #
    # basic_covariates = ['monthly_premium', 'policy_wait']
    # pss = prop.PropensityScoreSpecification(df, t, basic_covariates)
    # df['e'] = pss.specification_create().score()
    # df['t'] = t
    # joblib.dump(df, 'df.pkl')
