import sys
import json
import requests
import itertools
from bs4 import BeautifulSoup
from pymongo import MongoClient


end_of_book_list = ['Moroni 10', 'Doctrine and Covenants 138', 'Articles of Faith 1', 'Malachi 3', 'Revelation 22']
go = True

client = MongoClient()
db = client['scrip_footnotes']
# print(db.collection_names())

url = 'https://www.lds.org/scriptures/bofm/1-ne/1?lang=eng'
while go:
    try:
        r = requests.get(url)
    except:
        print('\nError, chapter\n')
        continue  # if the connection times out it will go to the beginning of the loop without updating...essentially, a second try
    soup = BeautifulSoup(r.text)
    current_chapter = soup.find('span', attrs={'class': 'active'}).text
    print(current_chapter)

    source = url.split('/')[-3]
    book = url.split('/')[-2]
    chapter = url.split('/')[-1].split('?')[0]

    tab = db[source]

    for verse in soup.find_all('p', attrs={'class': 'verse'}):
        current_verse = verse.span.text.strip()

        for footnote in verse.find_all('a', attrs={'class': 'footnote study-note-ref'}):
            letter = footnote.sup.text
            url_new = footnote['rel'][0]

            while True:
                try:
                    r_new = requests.get(url_new)
                except:
                    print('\nError, footnote')
                    print(current_verse)
                    print(letter)
                    print('\n')
                    continue
                break

            soup_new = BeautifulSoup(r_new.text)
            # footnote_list = '; '.join([a.text for a in soup_new.find_all('a')])  # old way
            footnote_list = soup_new.p.text.strip()  # new way

            tab.insert_one(
                {
                    'source': source,
                    'book': book,
                    'chapter': chapter,
                    'verse': current_verse,
                    'letter': letter,
                    'footnote': footnote_list
                }
            )

    if current_chapter in end_of_book_list:
        if current_chapter == end_of_book_list[0]:
            url = 'https://www.lds.org/scriptures/dc-testament/dc/1?lang=eng'
        elif current_chapter == end_of_book_list[1]:
            url = 'https://www.lds.org/scriptures/pgp/moses/1?lang=eng'
        elif current_chapter == end_of_book_list[2]:
            url = 'https://www.lds.org/scriptures/ot/gen/1?lang=eng'
        elif current_chapter == end_of_book_list[3]:
            url = 'https://www.lds.org/scriptures/nt/matt/1?lang=eng'
        else:
            go = False
    else:
        url_next = str(soup.find('li', attrs={'class': 'next'}).a['href'])
        if url_next.count('/') == 5:
            url = url_next.split('?')[0] + '/1'
        else:
            url = url_next

# show dbs
# show collections
# use <database name>
# use <collection name>
# db.collection_name.find()   shows all documents in this collections
# db.bofm.find({'book': '2-ne', 'chapter': '20'})
# db.bofm.deleteMany({'book': '2-ne', 'chapter': '21'})
# db.bofm.stats().count
# db['dc-testament'].stats().count