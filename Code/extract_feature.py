from utils.cp_test import get_core_neighbor, get_core_neighbor_test
from utils.network_fea import get_network_features
import os
import time
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


QUERIED_DATA_PATH = '../Data/queriedData/'
PROCESSED_DATA_PATH = '../Data/processedData/'


def main(args):
    # Data Cleaning
    token_name = args.token_name
    print('Data Clean')
    time_start = time.time()

    # Read data
    end_date = args.end_date        # default: 2022-07-13
    dtypes = {'token_address': str, 'from_address': str,
              'to_address': str, 'timestamp': str, 'value': np.float64}
    data = pd.read_csv(
        f"{QUERIED_DATA_PATH}{token_name}_{end_date}.csv", dtype=dtypes)

    # Filter out the transactions with 0x0000000000000000000000000000000000000000
    data = data[data['from_address'] !=
                '0x0000000000000000000000000000000000000000']
    data = data[data['to_address'] !=
                '0x0000000000000000000000000000000000000000']

    start_date = list(data['timestamp'].unique())[-1]
    end_date = list(data['timestamp'].unique())[0]

    # Aggregate tx values between the addresses pair on the same day
    agg_data = data.groupby(
        ['timestamp', 'from_address', 'to_address']).sum().reset_index()

    # Print info
    print(f'>> Token: {token_name}')
    print(f'>> Datetime range: from {start_date} to {end_date}')
    print(f'>> Number of transaction: {data.shape[0]}')
    print(
        f'>> Number of addresses involved: {len(set(list(data.from_address + data.to_address)))}')
    print(f">> Total value involved: {data['value'].sum()}")
    print(f'>> Time to clean data: {time.time()-time_start:.2f}s')
    print('=========================================================\n')

    # Network Features Extracting
    print('============ Network Features ============')
    time_start = time.time()
    # Get network features
    network_fea = agg_data.reset_index().groupby(
        'timestamp').apply(get_network_features).reset_index()
    network_fea.drop(columns=['level_1'], inplace=True)

    # Calculate top10_degree_ratio
    network_fea['top10_degree_ratio'] = network_fea['top10_degree_mean'] / \
        network_fea['degree_mean']

    # Rearrange columns
    cols = list(network_fea.columns)
    network_fea = network_fea[cols[:7] + [cols[-1]] + cols[7:-1]]

    # Print info
    print(f'>> Number of features: {network_fea.shape[1]}')
    print(
        f'>> Time to calculate network feature: {time.time()-time_start:.2f}s')
    print('=========================================================\n')

    # Core-Periphery Test
    print('Core-periphery Analysis')
    time_start = time.time()

    # Update nerwork_fea with num_core and avg_core_neighbor
    if args.test == 0:
        network_fea_t = agg_data.reset_index().groupby(
            'timestamp').apply(get_core_neighbor).reset_index()
        network_fea['num_core'] = network_fea_t['num_core']
        network_fea['avg_core_neighbor'] = network_fea_t['avg_core_neighbor']

    elif args.test == 1:
        network_fea_t = agg_data.reset_index().groupby(
            'timestamp').apply(get_core_neighbor_test).reset_index()
        network_fea['num_core'] = network_fea_t['num_core']
        network_fea['avg_core_neighbor'] = network_fea_t['avg_core_neighbor']
        network_fea['significance'] = network_fea_t['significance']

    # How many days that each address is a core
    core_addresses_list = [core_address for core_addresses in list(
        network_fea_t.core_addresses) for core_address in core_addresses]
    core_days_cnt = pd.Series(core_addresses_list).value_counts(
        ascending=False).reset_index()
    core_days_cnt.columns = ['address', 'core_days_cnt']

    # Print info
    print(f'>> Number of features: {network_fea.shape[1]}')
    print(
        f'>> Time for core-periphery analysis: {time.time()-time_start:.2f}s')
    print('=========================================================\n')

    # Data Saving
    print('Save Data')
    # Make dir
    SAVE_DIR = f"{PROCESSED_DATA_PATH}{token_name}_{args.end_date}"
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
    parser = argparse.ArgumentParser(description='BNS')

    # Token options
    parser.add_argument('--token-name', type=str, default='AAVE',
                        help='The name of the token to be processed')
    parser.add_argument('--start-date', type=str, help='Start date')
    parser.add_argument('--end-date', type=str,
                        default='2022-07-13', help='End date')
    parser.add_argument('--time-range', type=int, default=1, help='Time range')

    # CP Structure test
    parser.add_argument('--test', type=int, default=0,
                        help='0-do not conduct cp structure test; 1-conduct cp structure test')

    args = parser.parse_args()

    main(args)
