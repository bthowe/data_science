import sys
import joblib
import requests
from bs4 import BeautifulSoup

url = 'https://www.lds.org/scriptures/abbreviations?lang=eng'
r = requests.get(url)
soup = BeautifulSoup(r.text)

ab_lst = []
for tbody in soup.find_all('tbody'):
    for tr in tbody.find_all('tr'):
        ab_lst.append(tr.td.text.strip().replace('\xa0', ' '))
joblib.dump(ab_lst, 'ab_lst.pkl')
