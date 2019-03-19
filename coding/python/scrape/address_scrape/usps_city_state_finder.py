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


# 'https://www.melissadata.com/lookups/addresscheck.asp'  # looks like a potentially useful place for demographic data; can't do the volume I'd like for free, however
# it is not trivial to scrape a google website such as maps
# https://stackoverflow.com/questions/21641887/python-multiprocessing-process-hangs-on-join-for-large-queue  # amd's response is extremely helpful


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
    print(url); sys.exit()

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


def usps_lookup(mat):
    """mat is an n x 2 numpy array, with stree address and zipcode in columns one and two, respectively"""
    start = time.time()

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


def compare(df1, df2):
    df2['addressstreet1'] = df2['addressstreet1'].str.replace('#', '')
    df_failed = df2.\
        merge(df1, how='left', left_on=['addressstreet1', 'addresszip'], right_on=['original_address', 'original_zip'], indicator=True). \
        query('_merge == "left_only"')
    return df_failed[['addressstreet1', 'addresszip']].values


def main():
    # mat = joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/icims.pkl'). \
    #     query('addressstreet1 != ""') \
    #     [['addressstreet1', 'addresszip']].\
    #     values  #[:100, :]
    # usps_lookup(mat)

    df = compare(
        joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/address_city_state.pkl').drop_duplicates(subset=['original_address', 'original_zip'], keep='last'),
        joblib.load('/Users/travis.howe/PycharmProjects/sales_talent_acquisition/icims/data_files/icims.pkl').query('addressstreet1 != ""')
    )
    usps_lookup(df)
# Why can't I get these to render? Most work through the website UI


if __name__ == '__main__':
    main()




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


# 1. is it something with the usps server?
    # 1. look into the requests library and sending the request again.
    # 2. I don't know why it poops out on me...I think it might be the usps server.


'''
http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=
<ZipCodeLookupRequest USERID="924SPRIN3138">
    <Address ID="0"><Address1></Address1><Address2>809 Logan Ave.</Address2><City></City><State></State><Zip5>64012</Zip5></Address>
</ZipCodeLookupRequest>
'''
'''
http://production.shippingapis.com/ShippingAPITest.dll?API=ZipCodeLookup&XML=
<ZipCodeLookupRequest USERID="924SPRIN3138">
    <Address ID="0"><Address1></Address1><Address2>7671 monroe ave</Address2><City></City><State></State><Zip5>64132</Zip5></Address>
</ZipCodeLookupRequest>
'''

'''
    <Address ID="0"><Address1></Address1><Address2>One H&R Block Way</Address2><City></City><State></State><Zip5>64105</Zip5></Address>
    <Address ID="1"><Address1></Address1><Address2>4235 Heatherview Drive</Address2><City></City><State></State><Zip5>52302</Zip5></Address>
    <Address ID="2"><Address1></Address1><Address2>Calle 6 H-19</Address2><City></City><State></State><Zip5>00985</Zip5></Address>
    <Address ID="3"><Address1></Address1><Address2>Historic 18th & Vine Jazz District 1601 E 18th Street, 365</Address2><City></City><State></State><Zip5>64108</Zip5></Address>
    <Address ID="4"><Address1></Address1><Address2>7671 monroe ave</Address2><City></City><State></State><Zip5>64132</Zip5></Address>
'''
# works: 1, 2, 4
# todo: why don't 0 and 3 work? There have to be others that threw an error as well, no?