import sys
import time
from bs4 import BeautifulSoup
from requestium import Session, Keys
# https://github.com/tryolabs/requestium

def login():
    url = 'https://ident.lds.org/sso/UI/Login'
    url2 = 'https://www.lds.org/mls/mbr/records/member-list?lang=eng'

    import os
    print(os.getcwd())

    s = Session('/Users/travis.howe/Projects/github/data_science/scrape/email_lst_scrape/chromedriver', browser='chrome', default_timeout=15)
    s.driver.get(url)

    print('Waiting for elements to load...')
    s.driver.ensure_element_by_id('IDToken1').send_keys(Keys.BACKSPACE)
    s.driver.ensure_element_by_id('IDToken2').send_keys(Keys.BACKSPACE)
    # s.driver.ensure_element_by_id('IDToken1').send_keys(lds_user_name)
    # s.driver.ensure_element_by_id('IDToken2').send_keys(lds_password)
    print('Please log-in in the chrome browser')

    s.driver.ensure_element_by_id("login-submit-button", timeout=60, state='present').click()

    s.driver.get(url2)
    s.driver.ensure_element_by_tag_name("tbody", timeout=60, state='visible')

    # todo: this isn't great
    go = True
    email_lst = []
    while go:
        s.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(3)
        new_page = s.driver.page_source
        if 'Ziemann, Donella' in new_page:
            go = False
    soup = BeautifulSoup(new_page, 'lxml')
    email_lst += [href.split(':')[1] for href in [a_tag['href'] for a_tag in soup.findAll('a') if a_tag.has_attr('ng-href')] if '@' in href]
    return email_lst

def lst_to_txt_file(lst):
    with open ('/Users/travis.howe/Projects/github/data_science/scrape/email_lst_scrape/ward_email_lst.txt', 'w') as text_file:
        for email in set(lst):
            text_file.write('{}, '.format(email))

if __name__ == '__main__':
    lst_to_txt_file(login())
