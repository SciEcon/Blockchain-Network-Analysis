import numpy as np
import pandas as pd
import networkx as nx
import community


def get_network_features(x):
    # Construct daily tx graph
    G = nx.from_pandas_edgelist(
        x, 'from_address', 'to_address', 'value', nx.Graph())
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
