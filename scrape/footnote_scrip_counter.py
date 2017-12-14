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


client = MongoClient()
db = client['scrip_footnotes']

ab_lst = joblib.load('ab_lst.pkl')


class FootnoteParser(object):
    def __init__(self, footnote_str):
        self.footnote_str = footnote_str

    def _split_footnote(self, txt, seps):
        default_sep = seps[0]

        for sep in seps[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip()[:-1] for i in txt.split(default_sep)][1:]

    def _abs_in_footnote(self, txt, seps):
        return [sep for sep in seps if sep in txt]

    def references_add_prefix(self):
        prefixes = self._abs_in_footnote(self.footnote_str, ab_lst)
        suffixes = self._split_footnote(self.footnote_str, ab_lst)

        ref_lst = []
        for combos in itertools.product(prefixes, suffixes):
            prefix = combos[0]
            suffix = combos[1]

            ref = ' '.join(combos)
            if ref in self.footnote_str:
                ref_lst += ['{0} {1}'.format(prefix, s) for s in suffix.split('; ')]
        return ref_lst

    def references_flatten(self, lst):
        ref_lst = []
        for ref in lst:
            if '(' in lst:
                pass  #todo: (1) take the numbers between the parentheses, (2) split by comma, (3) generate list
            else:
                ref_lst.append(ref)

if __name__ == '__main__':
    # for document in db['bofm'].find({'book': '3-ne', 'chapter': '27'}):
    for document in db['nt'].find({'book': 'rev', 'chapter': '19'}):
        footnote = document['footnote'].replace('\xa0', ' ')
        fp = FootnoteParser(footnote)
        print(fp.references_add_prefix())

# todo: test this for other books, etc.
#   -the jst is not correct---it uses two keywords (i.e., "jst" and "rev" (for example))
# todo: polish the class---i.e., inputs, outputs, etc.