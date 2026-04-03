import numpy as np
import random
from environment import SocialNetwork
from agent import RLAgent

class SimulationEngine:
    """
    Spatial Evolutionary IPD engine combining:
    1. Nowak & May (1992) deterministic best-neighbor imitation
    2. Q-Learning for intelligent exploration
    
    Supports multiple network topologies for comparative analysis.
    """
    def __init__(self, n=100, k=6, p=0.0, alpha=0.3, gamma=0.5, epsilon=0.02,
                 T=1.4, R=1.0, P=0.1, S=0.0, init_defector_fraction=0.1,
                 graph_type="watts_strogatz"):
        self.env = SocialNetwork(n, k, p, graph_type=graph_type)
        self.agents = {}
        self.n = n
        
        for node in self.env.graph.nodes():
            self.agents[node] = RLAgent(node_id=node, alpha=alpha, gamma=gamma, epsilon=epsilon)
            
        self.T = T
        self.R = R
        self.P = P
        self.S = S
        
        self._initialize_strategies(init_defector_fraction)

    def _initialize_strategies(self, defector_fraction):
        """Start with mostly cooperators and seed a minority of defectors."""
        all_nodes = list(self.env.graph.nodes())
        num_defectors = max(1, int(len(all_nodes) * defector_fraction))
        
        for node in all_nodes:
            self.env.update_node_state(node, 1)
            self.agents[node].last_action = 1
            
        defector_nodes = random.sample(all_nodes, num_defectors)
        for node in defector_nodes:
            self.env.update_node_state(node, 0)
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

    def inject_defectors(self, count):
        """Inject defectors into a running simulation (for resilience testing)."""
        cooperators = [n for n in self.env.graph.nodes() if self.env.get_node_state(n) == 1]
        count = min(count, len(cooperators))
        targets = random.sample(cooperators, count)
        for node in targets:
            self.env.update_node_state(node, 0)
            self.agents[node].last_action = 0

    def step(self):
        """One synchronous round."""
        # 1. Determine actions
        actions = {}
        for node in self.agents:
            current_strategy = self.env.get_node_state(node)
            if random.random() < self.agents[node].epsilon:
                coop_r = self.get_coop_ratio(node)
                actions[node] = self.agents[node].choose_action(coop_r)
            else:
                actions[node] = current_strategy
                self.agents[node].last_action = current_strategy

        # 2. Calculate total payoff
        rewards = {}
        for node in self.agents:
            neighbors = self.env.get_neighbors(node)
            total = sum(self._payoff(actions[node], actions[n]) for n in neighbors)
            rewards[node] = total
            self.env.update_node_score(node, total)

        # 3. Q-Learning Update
        for node in self.agents:
            neighbors = self.env.get_neighbors(node)
            norm_reward = rewards[node] / max(len(neighbors), 1)
            next_coop = self.get_coop_ratio(node)
            self.agents[node].learn(norm_reward, next_coop)

        # 4. Spatial Imitation: copy the best in neighborhood
        new_strategies = {}
        for node in self.agents:
            neighbors = list(self.env.get_neighbors(node))
            candidates = [node] + neighbors
            best = max(candidates, key=lambda n: rewards[n])
            new_strategies[node] = actions[best]
        
        for node, strategy in new_strategies.items():
            self.env.update_node_state(node, strategy)
            self.agents[node].last_action = strategy

        return self.env.get_cooperation_rate(), rewards
