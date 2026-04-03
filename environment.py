import networkx as nx
import numpy as np

class SocialNetwork:
    """
    The world/environment: a graph where nodes are people and edges are relationships.
    Supports multiple network topologies used in social network research.
    """
    
    GRAPH_TYPES = {
        "watts_strogatz": "Small-World (Watts-Strogatz)",
        "barabasi_albert": "Scale-Free (Barabási-Albert)",
        "erdos_renyi": "Random (Erdős-Rényi)",
        "grid": "Regular Grid Lattice",
    }
    
    def __init__(self, n=100, k=4, p=0.1, graph_type="watts_strogatz"):
        self.n = n
        self.k = k
        self.p = p
        self.graph_type = graph_type
        self.graph = self._generate_graph()
        
    def _generate_graph(self):
        """Generates a graph based on the specified type and initializes node attributes."""
        if self.graph_type == "watts_strogatz":
            G = nx.watts_strogatz_graph(n=self.n, k=self.k, p=self.p)
        elif self.graph_type == "barabasi_albert":
            # m = number of edges to attach from a new node (use k//2 for comparable density)
            m = max(1, self.k // 2)
            G = nx.barabasi_albert_graph(n=self.n, m=m)
        elif self.graph_type == "erdos_renyi":
            # p_edge chosen so expected degree ≈ k
            p_edge = self.k / (self.n - 1)
            G = nx.erdos_renyi_graph(n=self.n, p=p_edge)
        elif self.graph_type == "grid":
            # Create a 2D grid with roughly n nodes
            side = int(np.ceil(np.sqrt(self.n)))
            G = nx.grid_2d_graph(side, side)
            # Relabel nodes to integers
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            # Trim to exactly n nodes
            nodes_to_remove = list(G.nodes())[self.n:]
            G.remove_nodes_from(nodes_to_remove)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        # Initialize node attributes
        for node in G.nodes():
            G.nodes[node]['state'] = np.random.choice([0, 1])
            G.nodes[node]['score'] = 0.0
            G.nodes[node]['round_reward'] = 0.0
            
        return G

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
        
    def get_node_state(self, node_id):
        return self.graph.nodes[node_id]['state']
        
    def update_node_state(self, node_id, new_state):
        self.graph.nodes[node_id]['state'] = new_state
        
    def update_node_score(self, node_id, reward):
        self.graph.nodes[node_id]['score'] += reward
        self.graph.nodes[node_id]['round_reward'] = reward

    def get_cooperation_rate(self):
        if self.n == 0:
            return 0
        cooperators = sum(1 for n in self.graph.nodes() if self.graph.nodes[n]['state'] == 1)
        return cooperators / len(self.graph.nodes())
    
    def get_scores(self):
        """Returns a list of all agent scores."""
        return [self.graph.nodes[n]['score'] for n in self.graph.nodes()]
    
    def get_states(self):
        """Returns a dict of {node_id: state}."""
        return {n: self.graph.nodes[n]['state'] for n in self.graph.nodes()}
    
    def get_graph_metrics(self):
        """Returns key network science metrics."""
        G = self.graph
        metrics = {}
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        
        # Average shortest path (only for connected graphs)
        if nx.is_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            # Use the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(subgraph) > 1:
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                metrics['avg_path_length'] = 0
                
        return metrics
