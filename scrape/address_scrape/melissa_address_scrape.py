import sys
import time
import joblib
import requests
import numpy as np
import pandas as pd
import multiprocessing as mp
from bs4 import BeautifulSoup
from requestium import Session, Keys

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




# url = 'https://www.melissadata.com/lookups/addresscheck.asp'


def login():

    import os
    print(os.getcwd())

    s = Session('/Users/travis.howe/Projects/github/data_science/scrape/address_scrape/chromedriver', browser='chrome', default_timeout=15)
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


def melissa_scrape():
    # can't do the volume I'd like for free
    add = '104 Wolsey Bridge Parkway'
    zip = '64012'
    add_str = '+'.join(add.split(' ')) + '+' + zip
    url = 'https://www.melissadata.com/lookups/AddressCheck.asp?LuAd=Melissa+Address+Table&email=&exprbox=&suites=&address=&address2=&city=&state=&zip=&dragbox={}'.format(add_str)

    page = requests.get(url).text
    print(page)
    sys.exit()
    soup = BeautifulSoup(page, 'lxml')
    t = soup.findAll('table')[2]

    for row in t.findAll('tr'):
        print(row)
        print('\n\n')

    sys.exit()


    # address, census entities


    sys.exit()


def usps_lookup():
    # must have city
    from pyusps import address_information

    addr = dict(
        [
            ('address', '104 Wolsey Bridge Parkway'),
            ('zip_code', '64012')
        ]
    )
    user_id = '924SPRIN3138'
    out = address_information.verify(user_id, addr)
    print(out)




def google_scrape():
    # doesn't allow bots...I'm not sure this is worth the workaround.
    import multiprocessing as mp
    num_process = 4

    mat = joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/icims.pkl'). \
        query('addressstreet1 != ""') \
        [['addressstreet1', 'addresszip']].\
        values

    mat_lst = np.array_split(mat[:16, :], num_process)

    output = mp.Queue()

    def _scraper(lst):
        full_address_lst = []
        for row in lst:
            # print(row)
            address = row[0]
            zip_code = row[1]
            # print(add, zip)
            # sys.exit()
            add_str = '+'.join(address.split(' ')) + '+' + zip_code
            url = 'https://www.google.com/maps/place/{}'.format(add_str)
            page = requests.get(url).text
            print(page)
            soup = BeautifulSoup(page, 'lxml')
            add = soup.find("meta", property="og:description")
            print(add)
            sys.exit()
            full_add = add['content']
            full_address_lst.append([address, zip_code, full_add])
            print(full_address_lst)
        output.put(full_address_lst)

    # for lst in mat_lst:
    #     print(lst.tolist())
    # sys.exit()

    processes = [mp.Process(target=_scraper, args=(lst.tolist(),)) for lst in mat_lst]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [output.get() for p in processes]
    print(results)




def usps_lookup():
    # amd's response is extremely helpful
    # https://stackoverflow.com/questions/21641887/python-multiprocessing-process-hangs-on-join-for-large-queue
    num_process = 4

    mat = joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/icims.pkl'). \
        query('addressstreet1 != ""') \
        [['addressstreet1', 'addresszip']].\
        values

    mat_lst = np.array_split(mat, num_process)
    # mat_lst = np.array_split(mat[:100, :], num_process)
    output = mp.Queue()

    def _scraper(lst):
        full_address_lst = []

        url = 'http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=<ZipCodeLookupRequest USERID="924SPRIN3138">'
        add_zip_temp = []
        for index, row in enumerate(lst):
            address = row[0]
            zip_code = row[1]
            if '#' in address:
                address = address.replace('#', '')
            add_zip_temp.append([address, zip_code])
            i = index % 5
            url += '<Address ID="{0}"><Address1></Address1><Address2>{1}</Address2><City></City><State></State><Zip5>{2}</Zip5></Address>'.format(i, address, zip_code)

            if ((index % 5 == 4) and (index >= 4)) or (index + 1 == len(lst)):
                print('still working...\n')
                url += '</ZipCodeLookupRequest>'
                page = requests.get(url).text
                soup = BeautifulSoup(page, 'lxml')
                for ind, out in enumerate(soup.findAll('address')):
                    if not out.findAll('error'):
                        full_address_lst.append(add_zip_temp[ind] + [out.address2.text, out.city.text, out.state.text, out.zip5.text, out.zip4.text])
                    else:
                        full_address_lst.append(add_zip_temp[ind] + ['', '', '', '', ''])

                url = 'http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=<ZipCodeLookupRequest USERID="924SPRIN3138">'
                add_zip_temp = []

        output.put(full_address_lst)

    processes = [mp.Process(target=_scraper, args=(lst.tolist(),)) for lst in mat_lst]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [output.get() for p in processes]

    df = pd.DataFrame()
    for frame in results:
        print(len(frame))
        df = df.append(pd.DataFrame(frame, columns=['original_address', 'original_zip', 'usps_address', 'city', 'state', 'zip5', 'zip4']))
    print(df.head())
    print(df.shape)
    joblib.dump(df, '/Users/travis.howe/Downloads/address_city_state.pkl')




def _scrape(lst, qout):
    print('still working...\n')
    full_address_lst = []
    add_zip_temp = []
    url = 'http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=<ZipCodeLookupRequest USERID="924SPRIN3138">'
    for index, row in enumerate(lst):
        address = row[0]
        zip_code = row[1]
        if '#' in address:
            address = address.replace('#', '')
        add_zip_temp.append([address, zip_code])
        i = index % 5
        url += '<Address ID="{0}"><Address1></Address1><Address2>{1}</Address2><City></City><State></State><Zip5>{2}</Zip5></Address>'.format(i, address, zip_code)
    url += '</ZipCodeLookupRequest>'
    try:
        page = requests.get(url).text
    except:
        print(url)
        sys.exit()
    soup = BeautifulSoup(page, 'lxml')
    for ind, out in enumerate(soup.findAll('address')):
        if not out.findAll('error'):
            full_address_lst.append(add_zip_temp[ind] + [out.address2.text, out.city.text, out.state.text, out.zip5.text, out.zip4.text])
        else:
            full_address_lst.append(add_zip_temp[ind] + ['', '', '', '', ''])
    qout.put(full_address_lst)


def usps_lookup():
    # amd's response is extremely helpful
    # https://stackoverflow.com/questions/21641887/python-multiprocessing-process-hangs-on-join-for-large-queue

    start = time.time()

    mat = joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/icims.pkl'). \
        query('addressstreet1 != ""') \
        [['addressstreet1', 'addresszip']].\
        values  #[:100, :]

    splits = int(np.ceil(mat.shape[0] / 5))

    mat_lst = np.array_split(mat, splits)
    processes = []
    output = []
    for chunk in mat_lst:
        qout = mp.Queue()
        worker = mp.Process(target=_scrape, args=(chunk, qout))
        worker.start()
        worker.join()
        processes.append(worker)
        output += qout.get()

        print(time.time() - start)

    df = pd.DataFrame(output, columns=['original_address', 'original_zip', 'usps_address', 'city', 'state', 'zip5', 'zip4'])
    print(df.head())
    print(df.shape)
    joblib.dump(df, '/Users/travis.howe/Downloads/address_city_state.pkl')



if __name__ == '__main__':
    # lst_to_txt_file(login())
    # melissa_scrape()

    usps_lookup()
    # google_scrape()

# bad
'''
http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=
<ZipCodeLookupRequest USERID="924SPRIN3138">'
    <Address ID="0"><Address1></Address1><Address2>1919 W 50th St.</Address2><City></City><State></State><Zip5>66205</Zip5></Address>
    <Address ID="1"><Address1></Address1><Address2>10412 E. 64th street</Address2><City></City><State></State><Zip5>64133</Zip5></Address>
    <Address ID="2"><Address1></Address1><Address2>6219 N. London Ave</Address2><City></City><State></State><Zip5>64151</Zip5></Address>
    <Address ID="3"><Address1></Address1><Address2>14081 W. Robinson St. Apt. 2406</Address2><City></City><State></State><Zip5>66223</Zip5></Address>
    <Address ID="4"><Address1></Address1><Address2>630 River Drive</Address2><City></City><State></State><Zip5>66111</Zip5></Address>
</ZipCodeLookupRequest>
'''

# goodvvvvv
''''''

'''
http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=
<ZipCodeLookupRequest USERID="924SPRIN3138">'
    <Address ID="0"><Address1></Address1><Address2>1919 W 50th St.</Address2><City></City><State></State><Zip5>66205</Zip5></Address>
    <Address ID="1"><Address1></Address1><Address2>10412 E. 64th street</Address2><City></City><State></State><Zip5>64133</Zip5></Address>
    <Address ID="2"><Address1></Address1><Address2>6219 N. London Ave</Address2><City></City><State></State><Zip5>64151</Zip5></Address>
    <Address ID="3"><Address1></Address1><Address2>14081 W. Robinson St. Apt. 2406</Address2><City></City><State></State><Zip5>66223</Zip5></Address>
    <Address ID="4"><Address1></Address1><Address2>1919 W 50th St.</Address2><City></City><State></State><Zip5>66205</Zip5></Address>
</ZipCodeLookupRequest>
'''

'''
http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=
<ZipCodeLookupRequest USERID="924SPRIN3138">
    <Address ID="0"><Address1></Address1><Address2>5223 Oak Leaf Dr Apt 17</Address2><City></City><State></State><Zip5>64129</Zip5></Address>
    <Address ID="1"><Address1></Address1><Address2>2400 NW Shady Bend Lane</Address2><City></City><State></State><Zip5>64081</Zip5></Address>
    <Address ID="2"><Address1></Address1><Address2>7608 E 108th Ter</Address2><City></City><State></State><Zip5>64134</Zip5></Address>
    <Address ID="3"><Address1></Address1><Address2>9725 NW 86th Terrace</Address2><City></City><State></State><Zip5>64153</Zip5></Address>  b bbb
</ZipCodeLookupRequest>
'''


# todo today
# 1. is it something with the usps server?
    # 1. look into the requests library and sending the request again.
    # 2. I don't know why it poops out on me...I think it might be the usps server.
# 2. global optimization
    # 1. (x)last day of the month
    # 2. how to make the allocation more normal at the end of the month:
    # problem: because of the time horizon, small daily perturbations become very large at the end of the month.
    # possible resolutions: (1) change the budget, (2) change the constraint type, (3) change the bounds, (4) change the functions
    # I'm leaning to (1): taken to the extreme, this can be no different than just using what was done at the beginning of the month. Seems like a balance between these two extremes
    # Allow some of the excess budget (defined using the initial optimization) to be used in the following days, but not all...maybe like 10%? 25%? This will allow excess to be used but gradually.
    # So we define the budget as (1) whatever the monthly was minus spend to this point, (2) sum of budgted spend to this point: (2) + (.25) * ((1) - (2))
    # I think we don't want just the straight percentage of excess because we'll run into the same problems that we are now. What if we taper this somewhat?
    # Maybe instead of a percentage, I should focus on a certain amount each day...other than weekends. How about no more than 10% of total average spend? This would be calculated by summing the weekdays and adding the maximum of this ten percent or whatever the excess is.
    # If we're under, I think we could do the same thing.
        # 0. the remaining budget was like 400k
        # 1. it is giving crazy allocations: 87923.46363077  86637.92005143 145746.34634862  95753.66426358
        # 2. using an inequality constraint gives: 18240.04843831 16954.50484753 76062.93201447 26070.24908066
# 3. berke and ci


