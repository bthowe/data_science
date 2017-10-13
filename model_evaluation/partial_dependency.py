import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

class PartialDependency(object):
    """Generates and plots the partial dependency function between the specificed feature and the probability of success
       in relative terms. Really, it's off the hook awesome."""
    def __init__(self, model, X, y, feature):
        self.model = model

        mask = self.model.best_estimator_.named_steps['select'].get_support()
        feature_list = np.array(self.model.best_estimator_.named_steps['feature_create'].transform(X).columns.tolist())[mask].tolist()
        self.X = pd.DataFrame(Pipeline(self.model.best_estimator_.steps[:-1]).transform(X), columns=feature_list)

        self.y = y
        self.feature = feature
        self.X_vals = self._values_to_iter()
        self.function = None

    def _values_to_iter(self):
        if len(self.X[self.feature].unique()) < 100:
            return self.X[self.feature].unique().tolist()
        else:
            return np.linspace(self.X[self.feature].quantile(0.05), self.X[self.feature].quantile(0.95), 100).tolist()

    def partial_dependency_function(self):
        X_ginormous = self.X.copy()
        X_ginormous[self.feature] = self.X_vals[0]
        for val in self.X_vals[1:]:
            X_temp = self.X.copy()
            X_temp[self.feature] = val
            X_ginormous = X_ginormous.append(X_temp)
        X_ginormous['predicted_target'] = self.model.best_estimator_.steps[-1][1].predict_proba(X_ginormous)[:, 1]
        self.function = X_ginormous.groupby(self.feature)['predicted_target'].mean().reset_index()
        return self.function

    def partial_dependency_plot(self):
        self.function['predicted_target'] = self.function['predicted_target'] - self.function['predicted_target'].iloc[0]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.function[self.feature], self.function['predicted_target'])
        ax.set_ylabel('Relative Probability')
        ax.set_xlabel('{0}'.format(self.feature))
        plt.show()


