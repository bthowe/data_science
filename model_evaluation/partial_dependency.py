import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy.signal import savgol_filter

class PartialDependency(object):
    """Generates and plots the partial dependency (PD) or idinvidual conditional expectation (ICE) function between the
    specified feature and the predicted outcome.

    Inputs:
        model: A function that takes a dataframe as input and outputs corresponding predictions
        X: The training data to pass into "model"
        feature: The feature of interest
    """
    def __init__(self, model, X, feature):
        # todo: extend to multiple features

        self.model = model
        self.X = X
        self.feature = feature
        self.X_vals = self._values_to_iter()
        self.X_predictions = self._predictions_data()

    def _values_to_iter(self):
        if len(self.X[self.feature].unique()) < 100:
            return self.X[self.feature].unique().tolist()
        else:
            return np.linspace(self.X[self.feature].quantile(0.05), self.X[self.feature].quantile(0.95), 100).tolist()

    def _predictions_data(self):
        X_ginormous = self.X.copy()
        X_ginormous[self.feature] = self.X_vals[0]
        for val in self.X_vals[1:]:
            X_temp = self.X.copy()
            X_temp[self.feature] = val
            X_ginormous = X_ginormous.append(X_temp)
        X_ginormous['predicted_target'] = self.model(X_ginormous)
        return X_ginormous

    def partial_dependency_function(self):
        return self.X_predictions.groupby(self.feature)['predicted_target'].mean().reset_index()

    def partial_dependency_plot(self):
        function = self.X_predictions.groupby(self.feature)['predicted_target'].mean().reset_index()
        function['predicted_target'] = function['predicted_target'] - function['predicted_target'].iloc[0]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(function[self.feature], function['predicted_target'])
        ax.set_ylabel('Relative Probability')
        ax.set_xlabel('{0}'.format(self.feature))
        plt.show()

    def ice_plot(self, sample=1, centered=False, include_PDP=False):
        """
        Creates an ICE plot.

        :param sample: The fraction of the total number of curves (i.e., the number of observations, N) to plot
        :param centered: A boolean indicating whether each curve should undergo a level shift to begin at 0
        :param include_PDP: A boolean indicating whether to include a plot of the PD curve
        """
        all_indeces = self.X_predictions.index.unique()
        frac = round(len(all_indeces) * sample)
        indeces = np.random.choice(all_indeces, size=frac, replace=False)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        for index in indeces:
            if centered:
                feature = self.X_predictions.loc[index][self.feature]
                outcome = self.X_predictions.loc[index]['predicted_target'] - self.X_predictions.loc[index].iloc[0]['predicted_target']
            else:
                feature = self.X_predictions.loc[index][self.feature]
                outcome = self.X_predictions.loc[index]['predicted_target']
            ax.plot(feature, outcome)

        if include_PDP:
            function = self.X_predictions.groupby(self.feature)['predicted_target'].mean().reset_index()
            function['predicted_target'] = function['predicted_target'] - function['predicted_target'].iloc[0]
            ax.plot(function[self.feature], function['predicted_target'], color='black')

        ax.set_ylabel('Relative Probability')
        ax.set_xlabel('{0}'.format(self.feature))
        plt.show()

    def d_ice_plot(self, sample=1):
        """
        This method creates plots of the partial derivative of the estimated response functions \hat{f}. When
        interactions between the feature of relevance and other covariates exist, the derivative lines will be
        heterogeneous. The standard deviation across these curves by feature value is also plotted, and serves as a
        measure of heterogeneity.

        :param sample: The fraction of the total number of curves (i.e., the number of observations, N) to plot
        """
        all_indeces = self.X_predictions.index.unique()
        frac = round(len(all_indeces) * sample)
        indeces = np.random.choice(all_indeces, size=frac, replace=False)

        Doutcome = []
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(2, 1, 1)
        for index in indeces:
            feature = self.X_predictions.loc[index][self.feature]
            outcome = savgol_filter(
                x=self.X_predictions.loc[index]['predicted_target'] - self.X_predictions.loc[index].iloc[0]['predicted_target'],
                window_length=5,
                polyorder=2
            )
            doutcome = np.gradient(outcome, feature)
            Doutcome.append(doutcome)
            ax.plot(feature, doutcome)
        ax.set_ylabel('Partial Derivative')
        ax.set_xlabel('{0}'.format(self.feature))

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(self.X_vals, np.std(Doutcome, axis=0))
        ax.set_ylabel('Standard Deviation of Partials')
        ax.set_xlabel('{0}'.format(self.feature))
        plt.show()

if __name__ == '__main__':
    N = 1000
    np.random.seed(2)
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['varepsilon'] = np.random.normal(size=(N, 1))
    df['Y'] = 0.2 * df['X1'] - 5 * df['X2'] + 10 * df['X2'] * np.where(df['X3'] >= 0, 1, 0) + df['varepsilon']

    X = df.drop('varepsilon', 1)
    y = X.pop('Y')

    xgb = XGBRegressor()
    xgb.fit(X, y)
    Y = lambda x: xgb.predict(x)

    pd = PartialDependency(Y, X, 'X2')
    # print(pd.partial_dependency_function())
    # pd.partial_dependency_plot()
    # pd.ice_plot(sample=.1, centered=True, include_PDP=True)
    pd.d_ice_plot(sample=.1)