import time
import numpy as np
import pandas as pd


class FeatureImportances(object):
    """For each feature used in a model, the method 'fi_distribution_generate' bootstraps a distribution for the
    difference between the metric value using the test data and the metric value using the test data where the values
    corresponding to a given feature for each observation are drawn with replacement from the distribution of values for
    this feature. The method 'positive_fi_list' returns the list of features and their corresponding means such that the
    0.05 quantiles (according to the distribution found in fi_distribution_generate) are positive. This method return a
    pandas series sorted by the mean difference in 'lift.'"""

    def __init__(self, model, X, y, metric, sample_size=1):
        self.model = model

        if 0 <= sample_size <= 1:
            frac = int(round(len(X) * sample_size))
            self.X = X.sample(frac=frac)
        else:
            self.X = X.sample(n=sample_size)
        self.y = y.loc[self.X.index]

        self.metric = metric

        self.reduction = {}

    def fi_distribution_generate(self, draws_num):
        start = time.time()
        for col in self.X.columns.tolist():
            print(col, time.time() - start)
            col_list = []
            for i in range(draws_num):
                X_rand = self.X.copy()
                X_rand[col] = X_rand[col].sample(frac=1, replace=True).values  # resample with replacement
                y_rand = self.model(X_rand.values)  # feature_names may not contain [, ] or <

                col_list.append(self.metric(self.y, self.model(self.X.values)) - self.metric(self.y, y_rand))  # feature_names may not contain [, ] or <
            self.reduction[col] = col_list
        return pd.DataFrame(self.reduction)

    def positive_fi_list(self):
        df = pd.DataFrame(self.reduction)
        quantiles = pd.DataFrame(df.quantile(q=0.05, axis=0))
        positive_effect = quantiles.loc[quantiles[0.050] > 0].index.tolist()  # generate list of features whose .05 quantile from distribution generated in previous method is greater than 0
        return df[positive_effect].mean().sort_values(ascending=False)
