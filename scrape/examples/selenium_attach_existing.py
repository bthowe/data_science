import time
from selenium import webdriver

driver = webdriver.Chrome('../email_lst_scrape/chromedriver')
url = driver.command_executor._url
session_id = driver.session_id

driver.get("http://www.gmail.com")

time.sleep(3)

while True:
    driver = webdriver.Remote(command_executor=url,desired_capabilities={})
    driver.session_id = session_id
    driver.get("http://www.gmail.com")
    time.sleep(3)
