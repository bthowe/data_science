import requests
import pandas as pd
from bs4 import BeautifulSoup

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def main():
    start_month = '9'
    start_day = '1'
    end_month = '9'
    end_day = '8'

    login_data = {'un': 'USERNAME', 'pw': 'PASSWORD'}
    df = pd.DataFrame()
    for year in map(str, range(2010, 2018)):
        print('Pulling data for {0} from {1}/{2} to {3}/{4}...'.format(year, start_month, start_day, end_month, end_day))
        form_data = 'INSPECT, NETWORK, CLICK ON FILE, FORM DATA IN HEADERS, VIEW SOURCE, REPLACE RELEVANT DATA FIELD'

        with requests.session() as s:
            s.get('https://na51.salesforce.com', params=login_data)
            data = requests.post(
                'https://na51.salesforce.com/REPORT NUMBER',
                headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
                cookies=s.cookies,
                data=form_data).content
        df = df.append(parse_data(data))
    return df.\
        dropna(subset=['Lead: System ID']).\
        reset_index(drop=True)

def parse_data(csv_data):
    soup = BeautifulSoup(csv_data, "lxml")
    data_lst = []
    for row in soup.findAll('tr'):
        data_lst.append([cell.text for cell in row.findAll('td')])

    columns = [
        "Lead: Lead Status",
        "Lead: System ID",
        "Monthly App Issued Count",
        "Lead: Dental Status",
        "Lead: Stips Agent",
        "INF Policy Applicant",
        "Lead: Lead ID",
        "Lead First Name",
        "Lead: Last Name",
        "Lead: Phone",
        "Lead: Email",
        "Effective Date Applicant",
        "Lead: Retention Status",
        "Spouse First Name",
        "Spouse Last Name",
        "Effective Date Spouse",
        "Lead: Retention Status Spouse",
        "Lead: Applicant State/Province",
        "Lead: Retention Agent",
        "Lead: Confirmation By",
        "Carrier Applicant"
    ]
    if len(data_lst) > 1:
        return pd.DataFrame(data_lst, columns=columns).reset_index(drop=True)
    return pd.DataFrame(columns=columns)


if __name__ == '__main__':
    df = main()
    print(df.info())
    print(df.head())
    print(df.tail())
