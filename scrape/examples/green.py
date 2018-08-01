import sys
import time
from selenium import webdriver

driver = webdriver.Chrome('../email_lst_scrape/chromedriver')
url = driver.command_executor._url
session_id = driver.session_id

driver.get("http://www.gmail.com")

time.sleep(90)

count = 1
while True:
    print(count)
    driver.get("http://www.gmail.com")
    time.sleep(300)
    count += 1
