from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

data = # todo: get mongo data
tfidf = TfidfVectorizer(input='content', stop_words='english')
df = tfidf.fit_transform(data)

NMF.fit_transform(df)


