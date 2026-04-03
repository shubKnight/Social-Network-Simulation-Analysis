import numpy as np
import networkx as nx

def gini_coefficient(values):
    """
    Calculates the Gini coefficient for a list of values.
    0 = perfect equality, 1 = perfect inequality.
    """
    values = np.array(values, dtype=float)
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def strategy_entropy(states_dict):
    """
    Shannon entropy of the strategy distribution.
    0 = all agents use the same strategy, 1 = perfectly mixed.
    """
    states = list(states_dict.values())
    n = len(states)
    if n == 0:
        return 0.0
    coop = sum(states)
    defect = n - coop
    probs = [coop / n, defect / n]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    # Normalize to [0, 1]
    return entropy


def cooperator_cluster_sizes(graph):
    """
    Returns a sorted list of cooperator cluster sizes (connected components
    of cooperating nodes). This reveals whether cooperators form defensive
    clusters or are scattered.
    """
    cooperator_nodes = [n for n in graph.nodes() if graph.nodes[n]['state'] == 1]
    if not cooperator_nodes:
        return []
    subgraph = graph.subgraph(cooperator_nodes)
    clusters = list(nx.connected_components(subgraph))
    sizes = sorted([len(c) for c in clusters], reverse=True)
    return sizes


def defector_cluster_sizes(graph):
    """Same but for defectors."""
    defector_nodes = [n for n in graph.nodes() if graph.nodes[n]['state'] == 0]
    if not defector_nodes:
        return []
    subgraph = graph.subgraph(defector_nodes)
    clusters = list(nx.connected_components(subgraph))
    sizes = sorted([len(c) for c in clusters], reverse=True)
    return sizes


def compute_all_metrics(env):
    """Compute a comprehensive metrics snapshot for the current simulation state."""
    scores = env.get_scores()
    states = env.get_states()
    graph_metrics = env.get_graph_metrics()
    
    coop_clusters = cooperator_cluster_sizes(env.graph)
    defect_clusters = defector_cluster_sizes(env.graph)
    
    return {
        'cooperation_rate': env.get_cooperation_rate(),
        'gini_coefficient': gini_coefficient(scores),
        'strategy_entropy': strategy_entropy(states),
        'num_cooperator_clusters': len(coop_clusters),
        'largest_cooperator_cluster': coop_clusters[0] if coop_clusters else 0,
        'num_defector_clusters': len(defect_clusters),
        'largest_defector_cluster': defect_clusters[0] if defect_clusters else 0,
        'avg_score': np.mean(scores) if scores else 0,
        'max_score': np.max(scores) if scores else 0,
        'min_score': np.min(scores) if scores else 0,
        **graph_metrics,
    }
