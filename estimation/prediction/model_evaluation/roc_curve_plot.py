import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def roc_plot(y_true, y_pred):
    def TPR(threshold):
        y = pd.concat(
            [
                y_true,
                y_pred.apply(lambda x: 1 if x > threshold else 0)
            ], axis=1
        ).rename(columns={'agg_target': 'y_true', 'ave_monthly_prob': 'y_pred'})
        y['TP'] = y.apply(lambda x: 1 if (x['y_true'] == 1) and (x['y_pred'] == 1) else 0, axis=1)
        y['FN'] = y.apply(lambda x: 1 if (x['y_true'] == 1) and (x['y_pred'] == 0) else 0, axis=1)
        return y['TP'].sum() / (y['TP'].sum() + y['FN'].sum())  # TPR = TP / (TP + FN)

    def FPR(threshold):
        y = pd.concat(
            [
                y_true,
                y_pred.apply(lambda x: 1 if x > threshold else 0)
            ], axis=1
        ).rename(columns={'agg_target': 'y_true', 'ave_monthly_prob': 'y_pred'})
        y['TN'] = y.apply(lambda x: 1 if (x['y_true'] == 0) and (x['y_pred'] == 0) else 0, axis=1)
        y['FP'] = y.apply(lambda x: 1 if (x['y_true'] == 0) and (x['y_pred'] == 1) else 0, axis=1)
        return 1 - (y['TN'].sum() / (y['TN'].sum() + y['FP'].sum()))  # TN / (TN + FP)

    df_roc = pd.DataFrame(np.linspace(0, 1, 101), columns=['threshold'])
    df_roc['TPR'] = df_roc['threshold'].apply(TPR)
    df_roc['FPR'] = df_roc['threshold'].apply(FPR)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_roc['FPR'], df_roc['TPR'], label="ROC Curve")
    ax.plot(df_roc['threshold'], df_roc['threshold'], color='r', label='Uninformative')
    ax.legend()
    plt.savefig('/Users/travis.howe/Downloads/roc_curve.png')
    plt.show()
