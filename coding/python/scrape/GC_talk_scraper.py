import sys
import json
import requests
import itertools
from bs4 import BeautifulSoup
from pymongo import MongoClient

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




# for tag in soup.find_all('a', attrs={'class': 'lumen-tile__link'}):
#         url = tag['href']
#         r = requests.get('https://www.lds.org/' + url)
#         soup = BeautifulSoup(r.text)
#
#         print(soup)
#         sys.exit()
#
#         talk_text = soup.find_all('div', attrs={'class': 'article-content'})[0].get_text().replace('\n', ' ')
#         speaker = tag.find_all('div', attrs={'class': 'lumen-tile__content'})[0].get_text()
#         title = tag.find_all('div', attrs={'class': 'lumen-tile__title'})[0].get_text().strip()
#         sys.exit()
#
#         tab.insert_one({'speaker': speaker, 'year': conference[0], 'month': conference[1], 'session': busi.rating,
#                         'talk_text': talk_text, 'title': title})
#
#
#         # todo: in csv file record text, name of speaker, year and month, session
