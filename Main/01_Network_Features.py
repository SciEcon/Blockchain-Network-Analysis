import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import community
import networkx as nx
import cpnet
import warnings 
warnings.filterwarnings('ignore')

from utils.network_fea import get_network_features
from utils.cp_test import get_core_neighbor

SAVE_PATH = '../Data/'

def main(args):
    ##################
    ### Data Clean ###
    ##################
    token_name = args.token_name
    print('============ Data Clean ============')
    time_start = time.time()
    
    # Read data
    today = '22-07-12' #date.today().strftime('%y-%m-%d')
    dtypes = {'token_address':str, 'from_address':str, 'to_address':str, 'timestamp':str, 'value':np.float64}
    data = pd.read_csv(f"../Data/{token_name}_{today}.csv", dtype=dtypes)
    start_date = list(data['timestamp'].unique())[-1]
    end_date = list(data['timestamp'].unique())[0]
    
    # Aggregate tx values between the addresses pair on the same day
    agg_data = data.groupby(['timestamp','from_address','to_address']).sum().reset_index()
    
    # Print info
    print(f'>> Token: {token_name}')
    print(f'>> Time range: from {start_date} to {end_date}')
    print(f'>> Number of tx: {data.shape[0]}')
    print(f'>> Number of addresses involved: {len(set(list(data.from_address + data.to_address)))}')
    print(f">> Total value involved: {data['value'].sum()}")
    print(f'>> Time to clean data: {time.time()-time_start:.2f}s')
    print('=========================================================\n')

    
    ########################
    ### Network Features ###
    ########################
    print('============ Network Features ============')
    time_start = time.time()
    # Get network features
    network_fea = agg_data.reset_index().groupby('timestamp').apply(get_network_features).reset_index()
    network_fea.drop(columns=['level_1'], inplace=True)

    # Calculate top10_degree_ratio
    network_fea['top10_degree_ratio'] = network_fea['top10_degree_mean'] / network_fea['degree_mean']

    # Rearrange columns
    cols = list(network_fea.columns)
    network_fea = network_fea[cols[:7] + [cols[-1]] + cols[7:-1]]

    # Print info
    print(f'>> Number of features: {network_fea.shape[1]}')
    print(f'>> Time to calculate network feature: {time.time()-time_start:.2f}s') 
    print('=========================================================\n')


    ###############################
    ### Core-periphery Analysis ###
    ###############################
    print('============ Core-periphery Analysis ============')
    time_start = time.time()

    # Update nerwork_fea with num_core and avg_core_neighbor
    network_fea_t = agg_data.reset_index().groupby('timestamp').apply(get_core_neighbor).reset_index()
    network_fea['num_core'] = network_fea_t['num_core']
    network_fea['avg_core_neighbor'] = network_fea_t['avg_core_neighbor']
    network_fea['significance'] = network_fea_t['significance']

    # How many days that each address is a core
    core_addresses_list = [core_address for core_addresses in list(network_fea_t.core_addresses) for core_address in core_addresses]
    core_days_cnt = pd.Series(core_addresses_list).value_counts(ascending=False).reset_index()
    core_days_cnt.columns = ['address', 'core_days_cnt']

    # Print info
    print(f'>> Number of features: {network_fea.shape[1]}')
    print(f'>> Time for core-periphery analysis: {time.time()-time_start:.2f}s') 
    print('=========================================================\n')


    #################
    ### Save Data ###
    #################
    print('============ Save Data ============')
    # Make dir
    SAVE_DIR = f"{SAVE_PATH}{token_name}_{today}"
    if os.path.exists(SAVE_DIR) is False:
        os.makedirs(SAVE_DIR)

    # Aggregated transaction data
    agg_data.to_csv(f"{SAVE_DIR}/01_agg_tx.csv", index=False)
    print(f"Saved > {SAVE_DIR}/01_agg_tx.csv")

    # Network features
    network_fea.to_csv(f"{SAVE_DIR}/02_network_fea.csv", index=False)
    print(f'Saved > {SAVE_DIR}/02_network_fea.csv')

    # Address core days
    core_days_cnt.to_csv(f'{SAVE_DIR}/03_core_address.csv', index=False)
    print(f'Saved > {SAVE_DIR}/03_core_address.csv')

    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BNS')

    # Token options
    parser.add_argument('--token-name', type=str, default='LQTY', help='Token name')
    parser.add_argument('--start-date', type=str, help='Start date')
    parser.add_argument('--time-range', type=int, default=1, help='Time range')

    args = parser.parse_args()

    main(args)