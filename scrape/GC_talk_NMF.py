# todo: setup python 3 environment
# todo: fold the other data in
# todo: what other variables can I use?
#   -Maybe speaker dummy
# todo: what about other years (I only extracted the last seven, or so)

import sys
import json
import requests
import itertools
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pymongo import MongoClient
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

client = MongoClient()
db = client['GC_talks']

X = []
for col in db.collection_names():
    collection = db[col]
    for doc in collection.find({}):
        X.append(doc['talk_text'])

print len(X)

v = TfidfVectorizer()
x = v.fit_transform(X)

# use pipeline, feature union

# nmf = NMF(n_components=10)
# nmf.fit_transform(X)
