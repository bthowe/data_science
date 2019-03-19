import sys
import time
from selenium import webdriver

driver = webdriver.Chrome('../email_lst_scrape/chromedriver')
url = driver.command_executor._url
session_id = driver.session_id

driver.get("http://www.google.com")

el=driver.find_elements_by_name('btnI')[0]

time.sleep(5)

action = webdriver.common.action_chains.ActionChains(driver)
action.move_to_element_with_offset(el, 5, 5)
action.click()  # not changing pages for some reason
action.perform()
