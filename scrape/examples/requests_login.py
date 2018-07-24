import os
import requests, lxml.html
from fake_useragent import UserAgent

url = 'https://ident.lds.org/sso/UI/Login'
url2 = 'https://www.lds.org/mls/mbr/records/member-list?lang=eng'

# this preserves a session, including cookies
with requests.Session() as s:
    login = s.get(url)

    # construct the correct form to post; can also do the following in the browser: (1) inspect, (2) Network, (3) click on correct item under "Name" (such as "Login"), and (4) under the "Headers" tab, scroll down to "Form Data"
    login_html = lxml.html.fromstring(login.text)
    hidden_inputs = login_html.xpath(r'//form//input[@type="hidden"]')
    form = {x.attrib["name"]: x.attrib["value"] for x in hidden_inputs}
    form['IDToken1'] = os.getenv('USERNAME')
    form['IDToken2'] = os.getenv('KEY')

    response = s.post(url, data=form)
    print(response.url)

    r = s.get(url2)
    print(r.text)

    # if need to pass a browser into the header, try
    ua = UserAgent()
    header = {'User-Agent': ua.chrome}
    s.get(url2, headers=header)
