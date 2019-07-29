import pandas as pd

df_aged = pd.DataFrame(columns=['aged', 'high_active', 'recovered'])
df_aged['aged'] = 20 * [1]
df_aged['high_active'] = 19 * [0] + [1]
df_aged['recovered'] = 14 * [1] + 5 * [0] + [1]

df_fresh = pd.DataFrame(columns=['aged', 'high_active', 'recovered'])
df_fresh['aged'] = 20 * [0]
df_fresh['high_active'] = 19 * [1] + [0]
df_fresh['recovered'] = 14 * [0] + 5 * [1] + [0]

df = df_aged.append(df_fresh).reset_index(drop=True)

print(df)

print('P(R = 1 | A = aged) = {}'.format(df.query('aged == 1')['recovered'].mean()))
print('P(R = 1 | A = fresh) = {}'.format(df.query('aged == 0')['recovered'].mean()))

print('P(I = 0 (low)| A = aged) = {}'.format(1 - df.query('aged == 1')['high_active'].mean()))
print('P(I = 0 (low) | A = fresh) = {}'.format(1 - df.query('aged == 0')['high_active'].mean()))

print('P(R = 1 | A = aged, I = low) = {}'.format(df.query('aged == 1').query('high_active == 0')['recovered'].mean()))
print('P(R = 1 | A = fresh, I = high) = {}'.format(df.query('aged == 0').query('high_active == 1')['recovered'].mean()))

py_do_aged = (df.query('aged == 1')['high_active'].mean()) * (df.query('aged == 1').query('high_active == 1')['recovered'].mean() * df['aged'].mean() + df.query('aged == 0').query('high_active == 1')['recovered'].mean() * (1 - df['aged'].mean())) + \
(1 - df.query('aged == 1')['high_active'].mean()) * (df.query('aged == 1').query('high_active == 0')['recovered'].mean() * df['aged'].mean() + df.query('aged == 0').query('high_active == 0')['recovered'].mean() * (1 - df['aged'].mean()))
print('P(R = 1 | do(A = aged)) = {}'.format(py_do_aged))

py_do_fresh = (df.query('aged == 0')['high_active'].mean()) * (df.query('aged == 1').query('high_active == 1')['recovered'].mean() * df['aged'].mean() + df.query('aged == 0').query('high_active == 1')['recovered'].mean() * (1 - df['aged'].mean())) + \
(1 - df.query('aged == 0')['high_active'].mean()) * (df.query('aged == 1').query('high_active == 0')['recovered'].mean() * df['aged'].mean() + df.query('aged == 0').query('high_active == 0')['recovered'].mean() * (1 - df['aged'].mean()))
print('P(R = 1 | do(A = fresh)) = {}'.format(py_do_fresh))

print('\nThus, the effect of a fresh bottle is {}'.format(py_do_fresh - py_do_aged))

# the key was making sure, whenever possible, a positive recovery was tied to a positive high_active and a zero recovery was tied to a zero high_active
