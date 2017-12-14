import sys
import json
import joblib
import requests
import itertools
from bs4 import BeautifulSoup
from pymongo import MongoClient

# simple counter of most tg topic, linked verse, etc.
# graph analysis of scrip network
# recommender for scrip; something that links to similar scrips
# some type of classification of chapters based on the footnotes



end_of_book_list = ['Moroni 10', 'Doctrine and Covenants 138', 'Articles of Faith 1', 'Malachi 4', 'Revelation 22']
go = True

client = MongoClient()
db = client['scrip_footnotes']

# for document in db.nt.find({'book': 'rev', 'chapter': '19'}):
    # for reference in document['footnote'].split('; '):


ab_lst = joblib.load('ab_lst.pkl')

references = list(db.nt.find({'book': 'rev', 'chapter': '18'}))[-1]['footnote'].replace('\xa0', ' ')
print(references)
print('\n')
book = ''
for reference in references.split('; '):
    rs = reference.split(' ')
    if rs[0] in ab_lst:
        book = rs[0]
        ref = reference
    else:
        ref = '{0} {1}'.format(book, rs[0])


    # if '(' in ref:
    #
    #     pass

    print(ref)


sys.exit()

footnote_lst = []
for collection in [db.bofm, db.dc-testament, db.pgp, db.ot, db.nt]:
    for document in collection.find():
        footnote = document['footnote'].replace('\xa0', ' ')
#         todo: parse footnote
#         todo: add to footnote_lst



# todo: are there instances in which semi-colons do not delimit the references?
#   -3 Nephi 27:21: There is a scrip followed by a TG reference, delimited by a period.
class FootnoteParser(object):
    def __init__(self, footnote_str):
        self.footnote_str = footnote_str

    def _prefix_set(self):
        # todo: I wonder if it would instead be better to split based on the keyword abbreviations.
        self.footnote_str.split('; ')