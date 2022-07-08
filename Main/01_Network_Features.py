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


def get_network_features(x):
    # Construct daily tx graph
    G = nx.from_pandas_edgelist(x, 'from_address', 'to_address', 'value', nx.Graph())
    result_dic = dict()
    
    # Number of nodes and edges
    num_edges = len(x['index'].unique())
    num_nodes = len(set(list(x['from_address'])+list(x['to_address'])))
    
    result_dic['num_edges'] = [num_edges]
    result_dic['num_nodes'] = [num_nodes]
    
    # Degree mean & std
    degree = list(dict(G.degree()).values())
    degree_mean = np.mean(degree)
    degree_std = np.std(degree)
    
    result_dic['degree_mean'] = [degree_mean]
    result_dic['degree_std'] = [degree_std]
    
    # Top 10 degree mean & std
    degree.sort(reverse=True)
    top_degree = degree[:10]
    top10_degree_mean = np.mean(top_degree)
    top10_degree_std = np.std(top_degree)
    
    result_dic['top10_degree_mean'] = [top10_degree_mean]
    result_dic['top10_degree_std'] = [top10_degree_std]
    
    # Degree centrality mean & std
    degree_centrality = list(nx.degree_centrality(G).values())
    degree_centrality_mean = np.mean(degree_centrality)
    degree_centrality_std = np.std(degree_centrality)
    
    result_dic['degree_centrality_mean'] = [degree_centrality_mean]
    result_dic['degree_centrality_std'] = [degree_centrality_std]
    
    # Modularity
    modularity = community.modularity(community.best_partition(G), G)
    result_dic['modularity'] = [modularity]
    
    # Transitivity
    transitivity = nx.transitivity(G)
    result_dic['transitivity'] = [transitivity]
    
    # Eigenvector centrality mean & std
    eig_cen = list(nx.eigenvector_centrality(G, max_iter=20000).values())
    eigenvector_centrality_mean = np.mean(eig_cen)
    eigenvector_centrality_atd = np.std(eig_cen)
    
    result_dic['eigenvector_centrality_mean'] = [eigenvector_centrality_mean]
    result_dic['eigenvector_centrality_atd'] = [eigenvector_centrality_atd]
    
    # Closeness centrality mean & std
    close_cen = list(nx.closeness_centrality(G).values())
    closeness_centrality_mean = np.mean(close_cen)
    closeness_centrality_std = np.std(close_cen)
    
    result_dic['closeness_centrality_mean'] = [closeness_centrality_mean]
    result_dic['closeness_centrality_std'] = [closeness_centrality_std]
    
    # Number of components
    num_components = nx.number_connected_components(G)
    result_dic['num_components'] = [num_components]
    
    # Size of gaint component / num of nodes
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    gaintG = G.subgraph(Gcc[0])
    giant_com_ratio = (1.0*gaintG.number_of_nodes())/G.number_of_nodes()
    result_dic['giant_com_ratio'] = [giant_com_ratio]
    
    return pd.DataFrame(result_dic)


def get_core_neighbor(tx_df):
    # Construct daily tx graph
    G = nx.from_pandas_edgelist(tx_df, 'from_address', 'to_address', 'value', nx.Graph())
    
    # Detecting discrete core-periphery structure
    alg = cpnet.BE()        # Load the Borgatti-Everett algorithm
    alg.detect(G)           # Feed the G as an input
    x = alg.get_coreness()  # Get the coreness of nodes
    c = alg.get_pair_id()   # Get the group membership of nodes
    
    # Calculate avg_core_neighbor
    core_addresses = [a for a in x if x[a]==1]               # core addresses
    degree = list(dict(G.degree(core_addresses)).values())   # get their degrees
    avg_core_neighbor = np.mean(degree)                      # average number of degree, i.e., avg_core_neighbor
    
    # p-value of significant test
    import warnings 
    warnings.filterwarnings('ignore')
    sig_c, sig_x, significant, p_values = cpnet.qstest(
            c, x, G, alg, significance_level=0.05, num_of_rand_net=100, num_of_thread=16)
    
    return pd.DataFrame({'num_core':[len(core_addresses)],
                         'avg_core_neighbor':[avg_core_neighbor],
                         'core_addresses':[core_addresses],
                         'significance':p_values})


def main():
    # ============ Data Clean ============
    token_name = 'LQTY'
    print(f'Token: {token_name}')
    
    print('============ Data Clean ============')
    time_start = time.time()
    # Read data
    dtypes = {'token_address':str, 'from_address':str, 'to_address':str, 'block_timestamp':str, 'f0_':np.float64}
    data = pd.read_csv('../Data/LQTY_2022-06-22.csv', dtype=dtypes)

    # Clean data
    data.rename(columns={'f0_':'value'}, inplace = True)
    data['timestamp'] = pd.to_datetime(data['block_timestamp']).dt.date
    data.drop(columns=['Unnamed: 0', 'token_address', 'block_timestamp'], inplace=True)
    data.sort_values(by='timestamp', ascending=False, inplace=True)

    # Time range
    start_date = list(data['timestamp'].unique())[-5]
    end_date = start_date + relativedelta(years=1)
    data = data[(data['timestamp']>=start_date) & (data['timestamp']<end_date)]

    # Print info
    print(f'>> Time range: from {start_date} to {end_date}')
    print(f'>> Number of tx: {data.shape[0]}')
    print(f'>> Number of addresses involved: {len(set(list(data.from_address + data.to_address)))}')
    print(f'>> Total value involved: {data.value.sum()}')

    # Save raw tx data
    SAVE_DIR = f"../Data/{token_name}_{start_date}-{end_date}"

    if os.path.exists(SAVE_DIR) is False:
        os.makedirs(SAVE_DIR)

    data.to_csv(f"{SAVE_DIR}/raw_tx.csv", index=False)
    print(f"Saved > {SAVE_DIR}/raw_tx.csv")

    # Aggregate tx values between the addresses pair on the same day
    data = data.groupby(['timestamp','from_address','to_address']).sum().reset_index()

    # Save aggregated tx data
    data.to_csv(f"{SAVE_DIR}/agg_tx.csv", index=False)
    print(f'Saved > {SAVE_DIR}/agg_tx.csv')
    
    time_end = time.time()
    print(f'\n> Data Cleaning Completed! {time_end-time_start} s\n')


    # ============ Network Analysis ============
    print('============ Network Analysis ============')
    time_start = time.time()
    # Get network features
    network_fea = data.reset_index().groupby('timestamp').apply(get_network_features).reset_index()
    network_fea.drop(columns=['level_1'], inplace=True)

    # Calculate top10_degree_ratio
    network_fea['top10_degree_ratio'] = network_fea['top10_degree_mean'] / network_fea['degree_mean']

    # Rearrange columns
    cols = list(network_fea.columns)
    network_fea = network_fea[cols[:7] + [cols[-1]] + cols[7:-1]]
    
    print('>> Network features calculated')
    print(f'>> Number of features: {network_fea.shape[1]}')
    
    time_end = time.time()
    print(f'\n> Network Analysis Completed! {time_end-time_start} s\n')
    
    
    # ============ Core-periphery Analysis ============
    print('============ Core-periphery Analysis ============')
    time_start = time.time()
    # Update nerwork_fea with num_core and avg_core_neighbor
    network_fea_t = data.reset_index().groupby('timestamp').apply(get_core_neighbor).reset_index()
    network_fea['num_core'] = network_fea_t['num_core']
    network_fea['avg_core_neighbor'] = network_fea_t['avg_core_neighbor']
    network_fea['significance'] = network_fea_t['significance']

    # Save network features
    network_fea.to_csv(f"{SAVE_DIR}/network_fea.csv", index=False)
    print(f'Saved > {SAVE_DIR}/network_fea.csv')

    # How many days that each address is a core
    core_addresses_list = [core_address for core_addresses in list(network_fea_t.core_addresses) for core_address in core_addresses]
    core_days_cnt = pd.Series(core_addresses_list).value_counts(ascending=False).reset_index()
    core_days_cnt.columns = ['address', 'core_days_cnt']
    # Save core_days_cnt
    core_days_cnt.to_csv(f'{SAVE_DIR}/core_days_cnt.csv', index=False)
    print(f'Saved > {SAVE_DIR}/core_days_cnt.csv')
    
    time_end = time.time()
    print(f'\n> Core-periphery Analysis Completed! {time_end-time_start} s')

    return



if __name__ == '__main__':
    main()