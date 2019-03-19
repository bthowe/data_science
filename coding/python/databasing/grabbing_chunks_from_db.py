import numpy as np
import pandas as pd

def _customers(id_list):
    query = '''
    SELECT *
    FROM customers
    WHERE customers.id IN {0};
    '''.format(id_list)
    return svg.get_data_from_mysql(unique_keyword_query=query)


df_customers = pd.DataFrame()
for l in np.array_split(df['customer_id'].values, 3):
    df_customers = df_customers.append(_customers(tuple(l)))
