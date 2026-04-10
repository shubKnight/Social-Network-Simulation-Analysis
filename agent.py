"""
Neural Agent — Wrapper for agent state in MADRL.
Each agent tracks its local state features to pass into the Shared PyTorch DQN.
"""

class NeuralAgent:
    def __init__(self, node_id, strategy=1, norm_degree=0.0, clustering=0.0):
        """
        Args:
            node_id: Node ID in the network graph.
            strategy: 1 = Cooperate, 0 = Defect (Initial).
            norm_degree: Degree of this node / Max degree in network.
            clustering: Local clustering coefficient.
        """
        self.node_id = node_id
        self.strategy = strategy
        self.round_payoff = 0.0
        self.cumulative_payoff = 0.0
        
        # Topological features (static for a given graph)
        self.norm_degree = norm_degree
        self.clustering = clustering

        # RL tracking
        self.last_state = None
        self.last_action = strategy

    def reset_round_payoff(self):
        self.round_payoff = 0.0
    
    def add_payoff(self, payoff):
        self.round_payoff += payoff
        self.cumulative_payoff += payoff

    def get_state(self, coop_neighbor_ratio):
        """Build the state vector for the Neural Network:
        [last_action, coop_neighbor_ratio, norm_degree, clustering]
        """
        return [float(self.last_action), float(coop_neighbor_ratio), 
                float(self.norm_degree), float(self.clustering)]
