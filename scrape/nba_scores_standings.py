import os
import datetime
import requests

my_dir = os.path.expanduser('~/Downloads/nba_daily')
today = str(datetime.date.today()).replace('-', '')
url = 'http://www.espn.com/nba/scoreboard/_/date/{}'.format(today)
r = requests.get(url)
with open(my_dir + '/nba_scoreboard.html', 'w') as my_file:
    my_file.write(r.text)

url = 'http://www.espn.com/nba/standings'
r = requests.get(url)
with open(my_dir + '/nba_standings.html', 'w') as my_file:
    my_file.write(r.text)
