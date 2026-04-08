"""
Evolutionary Agent — No learning, no Q-tables.
Each agent simply has a strategy (Cooperate or Defect) and accumulates payoff.
Strategy updates happen externally via the engine's imitation dynamics.
"""

class EvolutionaryAgent:
    def __init__(self, node_id, strategy=1):
        """
        Args:
            node_id: Node ID in the network graph.
            strategy: 1 = Cooperate, 0 = Defect.
        """
        self.node_id = node_id
        self.strategy = strategy
        self.round_payoff = 0.0        # Payoff from the current round
        self.cumulative_payoff = 0.0   # Total payoff across all rounds
    
    def reset_round_payoff(self):
        self.round_payoff = 0.0
    
    def add_payoff(self, payoff):
        self.round_payoff += payoff
        self.cumulative_payoff += payoff
    
    def get_normalized_payoff(self, degree):
        """Average payoff per interaction (normalizes for node degree)."""
        if degree == 0:
            return 0.0
        return self.round_payoff / degree
