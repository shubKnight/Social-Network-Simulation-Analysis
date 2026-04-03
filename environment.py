import networkx as nx
import numpy as np

class SocialNetwork:
    def __init__(self, n=100, k=4, p=0.1):
        """
        Initializes the Watts-Strogatz social network graph.
        
        Args:
            n (int): Number of nodes (agents)
            k (int): Each node is joined with its k nearest neighbors in a ring topology.
            p (float): The probability of rewiring each edge.
        """
        self.n = n
        self.k = k
        self.p = p
        self.graph = self._generate_graph()
        
    def _generate_graph(self):
        """Generates the Watts-Strogatz graph and initializes node attributes."""
        # Calculate Watts-Strogatz graph
        G = nx.watts_strogatz_graph(n=self.n, k=self.k, p=self.p)
        
        # Initialize node attributes
        for node in G.nodes():
            # State: 0 (Defector), 1 (Cooperator)
            # We initialize randomly or wait for agents to decide
            # For now, default randomly so there is an initial state
            G.nodes[node]['state'] = np.random.choice([0, 1])
            G.nodes[node]['score'] = 0.0
            G.nodes[node]['agent'] = None # Will hold the RLAgent instance
            
        return G

    def get_neighbors(self, node_id):
        """Returns a list of neighbor node IDs for a given node."""
        return list(self.graph.neighbors(node_id))
        
    def get_node_state(self, node_id):
        """Returns the current state (1 for Cooperator, 0 for Defector) of the node."""
        return self.graph.nodes[node_id]['state']
        
    def update_node_state(self, node_id, new_state):
        """Updates the state of a given node."""
        self.graph.nodes[node_id]['state'] = new_state
        
    def update_node_score(self, node_id, reward):
        """Adds to the cumulative score/wealth of a node."""
        self.graph.nodes[node_id]['score'] += reward

    def get_cooperation_rate(self):
        """Returns the global cooperation rate (percentage of cooperators)."""
        cooperators = sum(1 for n in self.graph.nodes() if self.graph.nodes[n]['state'] == 1)
        return cooperators / self.n if self.n > 0 else 0
