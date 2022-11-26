import numpy as np
import pandas as pd
import cpnet
import networkx as nx
import warnings 
warnings.filterwarnings('ignore')


def get_core_neighbor(tx_df):
    # Construct daily tx graph
    G = nx.from_pandas_edgelist(
        tx_df, 'from_address', 'to_address', 'value', nx.Graph())

    # Detecting discrete core-periphery structure
    alg = cpnet.MINRES()
    alg.detect(G)
    x = alg.get_coreness()

    # Calculate avg_core_neighbor
    core_addresses = [a for a in x if x[a] > 0.5]
    # get their degrees
    degree = list(dict(G.degree(core_addresses)).values())
    # average number of degree, i.e., avg_core_neighbor
    avg_core_neighbor = np.mean(degree)

    return pd.DataFrame({'num_core': [len(core_addresses)],
                         'avg_core_neighbor': [avg_core_neighbor],
                         'core_addresses': [core_addresses]})


def get_core_neighbor_test(tx_df):
    # Construct daily tx graph
    G = nx.from_pandas_edgelist(
        tx_df, 'from_address', 'to_address', 'value', nx.Graph())

    # Detecting discrete core-periphery structure
    alg = cpnet.MINRES()
    alg.detect(G)           # Feed the G as an input
    x = alg.get_coreness()  # Get the coreness of nodes
    c = alg.get_pair_id()   # Get the group membership of nodes

    # Calculate avg_core_neighbor
    core_addresses = [a for a in x if x[a] == 1]               # core addresses
    # get their degrees
    degree = list(dict(G.degree(core_addresses)).values())
    # average number of degree, i.e., avg_core_neighbor
    avg_core_neighbor = np.mean(degree)

    # p-value of significant test
    import warnings
    warnings.filterwarnings('ignore')
    sig_c, sig_x, significant, p_values = cpnet.qstest(
        c, x, G, alg, significance_level=0.05, num_of_rand_net=100, num_of_thread=16)

    return pd.DataFrame({'num_core': [len(core_addresses)],
                         'avg_core_neighbor': [avg_core_neighbor],
                         'core_addresses': [core_addresses],
                         'significance': [p_values]})
