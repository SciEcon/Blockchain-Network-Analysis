import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta


def filter_date(df, start_date, end_date):
    return df[(df['timestamp'] > start_date) & (df['timestamp'] < end_date)]

def count_unique_addresses(x):
    addresses = list(x['from_address']) + list(x['to_address'])
    return pd.DataFrame({'address_count': [len(set(addresses))]})

def sum_value(x):
    return pd.DataFrame({'value_sum': [sum(x['value'])]})