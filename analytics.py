import numpy as np
import networkx as nx
from agent import OCEAN_DIMS

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


# ── Personality-Aware Metrics ─────────────────────────────────────

def personality_assortativity(graph, agents):
    """
    Compute Newman's assortativity coefficient for each OCEAN dimension.
    Positive value = nodes with similar values on that dimension tend to
    be connected (personality-based echo chambers).
    Negative value = opposites attract.
    Near zero = no personality-based clustering.

    Uses the numeric assortativity coefficient from NetworkX.
    """
    result = {}
    for dim in OCEAN_DIMS:
        # Assign the personality value as a node attribute
        nx.set_node_attributes(graph,
            {n: agents[n].personality[dim] for n in agents},
            name=f'_pa_{dim}')
        try:
            r = nx.numeric_assortativity_coefficient(graph, f'_pa_{dim}')
            result[dim] = float(r) if not np.isnan(r) else 0.0
        except (ValueError, ZeroDivisionError):
            result[dim] = 0.0
    return result


def cooperation_by_personality(agents):
    """
    Average cooperation rate broken down by each OCEAN dimension,
    binned into Low (<0.33), Mid (0.33-0.67), and High (>0.67).
    Reveals which personality traits are most predictive of cooperation.
    """
    result = {}
    for dim in OCEAN_DIMS:
        bins = {'low': [], 'mid': [], 'high': []}
        for ag in agents.values():
            val = ag.personality[dim]
            trend = ag.strategy_trend
            if val < 0.33:
                bins['low'].append(trend)
            elif val > 0.67:
                bins['high'].append(trend)
            else:
                bins['mid'].append(trend)

        result[dim] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in bins.items()
        }
    return result


def personality_archetype_counts(agents):
    """
    Classify agents into personality archetypes based on their dominant
    personality traits and current behavior. Returns counts of each type.
    """
    archetypes = {
        'community_builder': 0,    # high-O, high-A, low-N, cooperating
        'strategic_hub': 0,        # high-E, high-C, cooperating
        'paranoid_isolationist': 0, # low-O, low-A, high-N
        'social_butterfly': 0,     # high-E, high-N
        'stoic_cooperator': 0,     # high-C, low-N, cooperating
        'opportunist': 0,          # low-A, low-C, defecting
        'other': 0,
    }

    for ag in agents.values():
        classified = False

        # Community Builder: open, agreeable, stable, cooperating
        if (ag.openness > 0.6 and ag.agreeableness > 0.6
                and ag.neuroticism < 0.4 and ag.strategy_trend > 0.6):
            archetypes['community_builder'] += 1
            classified = True

        # Strategic Hub: extraverted, disciplined, cooperating
        elif (ag.extraversion > 0.6 and ag.conscientiousness > 0.6
              and ag.strategy_trend > 0.5):
            archetypes['strategic_hub'] += 1
            classified = True

        # Paranoid Isolationist: closed, disagreeable, neurotic
        elif (ag.openness < 0.4 and ag.agreeableness < 0.4
              and ag.neuroticism > 0.6):
            archetypes['paranoid_isolationist'] += 1
            classified = True

        # Social Butterfly: extraverted, neurotic (reactive hub)
        elif ag.extraversion > 0.6 and ag.neuroticism > 0.6:
            archetypes['social_butterfly'] += 1
            classified = True

        # Stoic Cooperator: disciplined, stable, cooperating
        elif (ag.conscientiousness > 0.6 and ag.neuroticism < 0.4
              and ag.strategy_trend > 0.6):
            archetypes['stoic_cooperator'] += 1
            classified = True

        # Opportunist: disagreeable, impulsive, defecting
        elif (ag.agreeableness < 0.4 and ag.conscientiousness < 0.4
              and ag.strategy_trend < 0.4):
            archetypes['opportunist'] += 1
            classified = True

        if not classified:
            archetypes['other'] += 1

    return archetypes


def compute_all_metrics(env, agents=None):
    """Compute a comprehensive metrics snapshot for the current simulation state."""
    scores = env.get_scores()
    states = env.get_states()
    graph_metrics = env.get_graph_metrics()

    coop_clusters = cooperator_cluster_sizes(env.graph)
    defect_clusters = defector_cluster_sizes(env.graph)

    result = {
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

    # Personality metrics (if agents provided)
    if agents is not None:
        result['personality_assortativity'] = personality_assortativity(
            env.graph, agents)
        result['coop_by_personality'] = cooperation_by_personality(agents)
        result['personality_archetypes'] = personality_archetype_counts(agents)

    return result
