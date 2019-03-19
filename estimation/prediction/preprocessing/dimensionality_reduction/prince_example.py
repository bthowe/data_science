import pandas as pd
import prince

X = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data')
X.columns = ['Color', 'Size', 'Action', 'Age', 'Inflated']

print(X.head())


# m = MCA(
#     n_components=2,
#     n_iter=3,
#     copy=True,
#     check_input=True,
#     engine='auto',
#     random_state=42
# )
# m = m.fit(X)
# print(m.transform(X).head())

