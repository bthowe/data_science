import os
import datetime
import requests

my_dir = os.path.expanduser('~/Downloads/nba_daily')
today = str(datetime.date.today() - datetime.timedelta(days=1)).replace('-', '')
url_dict = {
    '/nba_scoreboard.html': 'http://www.espn.com/nba/scoreboard/_/date/{}'.format(today),
    '/nba_standings.html': 'http://www.espn.com/nba/standings',
    '/zach_lowe_articles.html': 'https://muckrack.com/zachlowe_nba/articles',
    '/zach_lowe_podcast.html': 'http://www.espn.com/espnradio/podcast/archive/_/id/10528553'
}

for filename, url in url_dict.items():
    r = requests.get(url)
    with open(my_dir + filename, 'w') as my_file:
        my_file.write(r.text)
