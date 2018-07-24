import sys
import time
from bs4 import BeautifulSoup
from requestium import Session, Keys

# https://github.com/tryolabs/requestium

lds_user_name = 'b.travis.howe'
url = 'https://ident.lds.org/sso/UI/Login'
url2 = 'https://www.lds.org/mls/mbr/records/member-list?lang=eng'

s = Session('/Users/travis.howe/Downloads/chromedriver', browser='chrome', default_timeout=15)
s.driver.get(url)

print('Waiting for elements to load...')
if lds_user_name:
    s.driver.ensure_element_by_id('IDToken1').send_keys(lds_user_name)
    s.driver.ensure_element_by_id('IDToken2').send_keys('M0ntana1m0ntana!')
    # s.driver.ensure_element_by_id('IDToken2').send_keys(Keys.BACKSPACE)
# print('Please log-in in the chrome browser')

s.driver.ensure_element_by_id("login-submit-button", timeout=60, state='present').click()

s.driver.get(url2)
s.driver.ensure_element_by_tag_name("tbody", timeout=60, state='visible')
# s.driver.ensure_element_by_class_name("pageTitle ng-scope", timeout=60, state='invisible')

go = True
email_lst = []
while go:
    s.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_page = s.driver.page_source
    if 'Ziemann, Donella' in new_page:
        go = False
    soup = BeautifulSoup(new_page)
    email_lst += [href.split(':')[1] for href in [a_tag['href'] for a_tag in soup.findAll('a') if a_tag.has_attr('ng-href')] if '@' in href]
print(email_lst)


# todo: I need to make sure these are the email addresses I want to use
# todo: put credentials somewhere safe





# pass browser into header
# selenium reference browser driver
# sessions in request
# using lxml.html to get the payload


# todo: move the chrome web driver into the folder in data_science