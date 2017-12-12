import sys
import json
import requests
import itertools
from bs4 import BeautifulSoup
from pymongo import MongoClient





# todo: stash in mongodb








end_of_book_list = ['Moroni 10', 'Doctrine and Covenants 138', 'Articles of Faith 1', 'Malachi 4', 'Revelation 22']
go = True

url = 'https://www.lds.org/scriptures/bofm/1-ne/1?lang=eng'
while go:
    r = requests.get(url)
    soup = BeautifulSoup(r.text)
    current_chapter = soup.find('span', attrs={'class': 'active'}).text
    print(current_chapter)


    for verse in soup.find_all('p', attrs={'class': 'verse'}):
        current_verse = verse.span.text
        print('\n\n\n')
        print(current_verse)

        for footnote in verse.find_all('a', attrs={'class': 'footnote study-note-ref'}):
            current_letter = footnote.sup.text
            print(current_letter)
            url_new = footnote['rel'][0]
            r_new = requests.get(url_new)
            soup_new = BeautifulSoup(r_new.text)
            footnote_list = '; '.join([a.text for a in soup_new.find_all('a')])
            print(footnote_list)

        sys.exit()
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



sys.exit()
client = MongoClient()
db = client['GC_talks']
for conference in itertools.product(range(2010, 2017), ['04', '10']):
    # db['year_{0}_month_{1}'.format(conference[0], conference[1])].drop()
    tab = db['year_{0}_month_{1}'.format(conference[0], conference[1])]

    r = requests.get('https://www.lds.org/general-conference/{0}/{1}?lang=eng'.format(conference[0], conference[1]))
    soup = BeautifulSoup(r.text)

    for section in soup.find_all('div', attrs={'class': 'section tile-wrapper layout--3 lumen-layout__item'})[:6]:
        session = section.find_all('span', attrs={'class': 'section__header__title'})[0].get_text()
        for tag in section.find_all('a', attrs={'class': 'lumen-tile__link'}):
            url = tag['href']
            r = requests.get('https://www.lds.org/' + url)
            soup = BeautifulSoup(r.text)

            talk_text = soup.find_all('div', attrs={'class': 'article-content'})[0].get_text().replace('\n', ' ')
            speaker = tag.find_all('div', attrs={'class': 'lumen-tile__content'})[0].get_text()
            title = tag.find_all('div', attrs={'class': 'lumen-tile__title'})[0].get_text().strip()

            print('{2}, {0}: {1}'.format(speaker, title, session))
            tab.insert_one({'speaker': speaker, 'year': conference[0], 'month': conference[1], 'session': session,
                            'talk_text': talk_text, 'title': title})

