import numpy as np
import random
from environment import SocialNetwork
from agent import RLAgent

class SimulationEngine:
    """
    Spatial Evolutionary IPD on Watts-Strogatz networks.
    
    Strategy update: Pure spatial imitation (Nowak & May 1992).
    Each agent copies the strategy of the highest-earning entity
    in their local neighborhood (including themselves).
    
    Q-Learning is used ONLY for occasional exploration — agents sometimes
    try a different strategy based on their learned Q-values.
    The exploration rate (epsilon) controls how often this happens.
    """
    def __init__(self, n=100, k=4, p=0.1, alpha=0.3, gamma=0.5, epsilon=0.05,
                 T=1.4, R=1.0, P=0.1, S=0.0, imitation_strength=10.0,
                 init_defector_fraction=0.1):
        self.env = SocialNetwork(n, k, p)
        self.agents = {}
        self.imitation_strength = imitation_strength
        self.n = n
        
        for node in self.env.graph.nodes():
            self.agents[node] = RLAgent(node_id=node, alpha=alpha, gamma=gamma, epsilon=epsilon)
            
        self.T = T
        self.R = R
        self.P = P
        self.S = S
        
        # Initialize: majority cooperators, seed a small cluster of defectors
        self._initialize_strategies(init_defector_fraction)

    def _initialize_strategies(self, defector_fraction):
        """
        Start with mostly cooperators and seed a minority of defectors.
        This mirrors the classic setup: cooperation is the default,
        and we observe whether defection can invade (and how topology affects this).
        """
        all_nodes = list(self.env.graph.nodes())
        num_defectors = max(1, int(len(all_nodes) * defector_fraction))
        
        # All start as cooperators
        for node in all_nodes:
            self.env.update_node_state(node, 1)  # Cooperate
            self.agents[node].last_action = 1
            
        # Seed defectors randomly
        defector_nodes = random.sample(all_nodes, num_defectors)
        for node in defector_nodes:
            self.env.update_node_state(node, 0)  # Defect
            self.agents[node].last_action = 0

    def get_coop_ratio(self, node_id):
        neighbors = self.env.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        return sum(1 for n in neighbors if self.env.get_node_state(n) == 1) / len(neighbors)

    def _payoff(self, a1, a2):
        if a1 == 1 and a2 == 1: return self.R
        if a1 == 1 and a2 == 0: return self.S
        if a1 == 0 and a2 == 1: return self.T
        return self.P

    def step(self):
        """One synchronous round."""
        
        # ── 1. Current strategies (from previous imitation result) ──
        # Most of the time, agents use their inherited strategy.
        # With probability epsilon, they explore via Q-learning.
        actions = {}
        for node in self.agents:
            current_strategy = self.env.get_node_state(node)
            
            if random.random() < self.agents[node].epsilon:
                # Explore: use Q-learning to pick action
                coop_r = self.get_coop_ratio(node)
                actions[node] = self.agents[node].choose_action(coop_r)
            else:
                # Exploit: use inherited strategy from imitation
                actions[node] = current_strategy
                self.agents[node].last_action = current_strategy

        # ── 2. Calculate total payoff ──
        rewards = {}
        for node in self.agents:
            neighbors = self.env.get_neighbors(node)
            total = sum(self._payoff(actions[node], actions[n]) for n in neighbors)
            rewards[node] = total
            self.env.update_node_score(node, total)

        # ── 3. Q-Learning Update (background learning) ──
        for node in self.agents:
            neighbors = self.env.get_neighbors(node)
            norm_reward = rewards[node] / max(len(neighbors), 1)
            next_coop = self.get_coop_ratio(node)
            self.agents[node].learn(norm_reward, next_coop)

        # ── 4. Spatial Imitation: copy the best in neighborhood ──
        new_strategies = {}
        for node in self.agents:
            neighbors = list(self.env.get_neighbors(node))
            candidates = [node] + neighbors
            best = max(candidates, key=lambda n: rewards[n])
            new_strategies[node] = actions[best]
        
        # Apply synchronously
        for node, strategy in new_strategies.items():
            self.env.update_node_state(node, strategy)
            self.agents[node].last_action = strategy

        return self.env.get_cooperation_rate(), rewards
