import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [
    [16, 48, 100],
    [14, 47, 92],
    [16, 45, 88],
    [12, 45, 95],
    [18, 46, 98],
    [18, 46, 101],
    [13, 47, 97],
    [16, 48, 98],
    [18, 49, 110],
    [22, 49, 124],
    [18, 50, 102],
    [19, 51, 115],
    [16, 52, 92],
    [16, 52, 102],
    [22, 50, 104],
    [12, 51, 85],
    [20, 54, 118],
    [14, 53, 105],
    [21, 52, 111],
    [17, 53, 122],
]

df = pd.DataFrame(data, columns=['X', 'W', 'Y'])


b = df.cov().loc['X', 'Y'] / df.cov().loc['X', 'X']
a = df['Y'].mean() - b * df['X'].mean()
print('1: a={0}, b={1}'.format(a, b))


print('\n')
def c(df, cols):
    """centering specificed columns"""
    for col in cols:
        df[col] = df[col] - df[col].mean()
    return df

df_c = df.copy().pipe(c, ['X'])
b_c = df_c.cov().loc['X', 'Y'] / df_c.cov().loc['X', 'X']
a_c = df_c['Y'].mean() - b_c * df_c['X'].mean()
print('2: a_c={0}, b_c={1}'.format(a_c, b_c))


print('\n')
df['Y_pred'] = df['X'] * b + a
df['Y_resid'] = df['Y'] - df['Y_pred']
print('3a: Is the variance of Y equal to the variance of Y hat plus the variance of the residuals? {}'.format(round(df.cov().loc['Y', 'Y'], 5) == round(df.cov().loc['Y_pred', 'Y_pred'] + df.cov().loc['Y_resid', 'Y_resid'], 5)))
print('3b: Is the Pearson correlation coefficient for X and Y equal to the ratio of variances of Y hat and Y? {}'.format(round(df.corr().loc['X', 'Y'] ** 2, 5) == round(df.cov().loc['Y_pred', 'Y_pred'] / df.cov().loc['Y', 'Y'], 5)))



print('\n')
def beta_weight(df, on_col, off_col):
    return (df.corr().loc[on_col, 'Y'] - df.corr().loc[off_col, 'Y'] * df.corr().loc[on_col, off_col]) / (1 - df.corr().loc[on_col, off_col] ** 2)

beta_x = beta_weight(df, 'X', 'W')
beta_w = beta_weight(df, 'W', 'X')
b_x = beta_x * (df['Y'].std() / df['X'].std())
b_w = beta_w * (df['Y'].std() / df['W'].std())
a = df['Y'].mean() - b_x * df['X'].mean() - b_w * df['W'].mean()
print('4a: a={0}, b_x={1}, b_w={2}'.format(a, b_x, b_w))

def std(df):
    return (df - df.mean().values) / df.std().values
    # print(df.std().values)

df_std = df.pipe(std)
beta_std_x = beta_weight(df_std, 'X', 'W')
beta_std_w = beta_weight(df_std, 'W', 'X')
b_std_x = beta_std_x * (df_std['Y'].std() / df_std['X'].std())
b_std_w = beta_std_w * (df_std['Y'].std() / df_std['W'].std())
a_std = df_std['Y'].mean() - b_std_x * df_std['X'].mean() - b_std_w * df_std['W'].mean()
print('4b: a_std={0}, b_std_x={1}, b_std_w={2}'.format(a_std, b_std_x, b_std_w))

df['Y_pred'] = a + b_x * df['X'] + b_w * df['W']
R2 = df.corr().loc['Y_pred', 'Y'] ** 2
print('4c: The overall multiple correlation: {}'.format(R2))
print('\t...ratio of ESS and TSS gives the same answer: {}'.format(((df['Y_pred'] - df['Y'].mean())**2).sum() / ((df['Y'] - df['Y'].mean())**2).sum()))


print('\n')
print('5: Adjusted R2: {}'.format(1 - (1 - R2) * ((19 / 17))))


print('\n')
print('6: histogram of residuals...')
df['Y_resid'] = df['Y'] - df['Y_pred']
df['Y_resid'].hist(bins=10)
# plt.show()


print('\n')
def first_order_partial_corr(df, on_col, off_col):
    return (df.corr().loc[on_col, 'Y'] - df.corr().loc[on_col, off_col] * df.corr().loc[off_col, 'Y']) / np.sqrt((1 - df.corr().loc[on_col, off_col] ** 2) * (1 - df.corr().loc[off_col, 'Y'] ** 2))
print('7a: first-order partial correlation between W and Y, removing X: {}'.format(first_order_partial_corr(df, 'W', 'X') ** 2))
def first_order_part_corr(df, on_col, off_col):
    return (df.corr().loc[on_col, 'Y'] - df.corr().loc[on_col, off_col] * df.corr().loc[off_col, 'Y']) / np.sqrt((1 - df.corr().loc[on_col, off_col] ** 2))
print('7a: first-order part correlation between W and Y, removing X: {}'.format(first_order_part_corr(df, 'W', 'X') ** 2))
# The former is correlation between W and Y after we take X's influence on both of these into account.
# THe latter is correlation between W and Y after we take into account X's influence only on W.
# From the book: "Variable W uniquely explains about 10.5% of the total variance in Y, and of the variance in Y not already
#   explained by X, predictor W accounts for about 19.9% of the rest."
# So the former looks at how much of the leftover variation in Y after accounting for X is uniquely explained by W, and the latter
#   looks at how much of the variation in Y is uniquely explained by W. So the latter should be smaller than the first since
#   it is relatively to all variation in Y whereas the former is relatively to that which is left over after accounting for X.
