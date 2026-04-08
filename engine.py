"""
Evolutionary Game Theory Engine — Spatial Imitation Dynamics.

Two update rules available:
1. "best_neighbor" (Nowak & May 1992): Copy the strategy of whoever earned
   the most in your neighborhood, including yourself. Deterministic, synchronous.
2. "fermi" (Santos & Pacheco 2005): Pick a random neighbor on each step and
   stochastically copy them based on payoff difference. Asynchronous.

References:
- Nowak, M.A. & May, R.M. (1992). Evolutionary games and spatial chaos. Nature.
- Santos, F.C. & Pacheco, J.M. (2005). Scale-free networks provide a unifying
  framework for the emergence of cooperation. PRL.
"""

import numpy as np
import random
from environment import SocialNetwork
from agent import EvolutionaryAgent


class SimulationEngine:
    def __init__(self, n=100, k=6, p=0.0,
                 T=1.3, R=1.0, P=0.0, S=0.0,
                 beta=10.0,
                 mutation_rate=0.005,
                 init_coop_fraction=0.8,
                 graph_type="watts_strogatz",
                 update_rule="best_neighbor",
                 rounds_per_update=5):
        """
        Args:
            n, k, p: Network parameters.
            T, R, P, S: Prisoner's Dilemma payoff matrix (must satisfy T > R > P >= S).
            beta: Fermi selection intensity (only used with 'fermi' update_rule).
            mutation_rate: Probability that an agent spontaneously defects each gen.
            init_coop_fraction: Initial fraction of cooperators (0.8 = invasion scenario).
            graph_type: watts_strogatz, barabasi_albert, erdos_renyi, grid.
            update_rule: "best_neighbor" or "fermi".
            rounds_per_update: Number of game rounds played before each strategy update.
                               Higher values = more stable payoff signal = less noise.
                               This is the KEY parameter for eliminating bimodality.
        """
        self.env = SocialNetwork(n, k, p, graph_type=graph_type)
        self.n = n
        self.beta = beta
        self.mutation_rate = mutation_rate
        self.update_rule = update_rule
        self.rounds_per_update = rounds_per_update
        
        self.T = T
        self.R = R
        self.P = P
        self.S = S
        
        # Create agents
        self.agents = {}
        all_nodes = list(self.env.graph.nodes())
        num_coop = int(len(all_nodes) * init_coop_fraction)
        coop_nodes = set(random.sample(all_nodes, num_coop))
        
        for node in all_nodes:
            strategy = 1 if node in coop_nodes else 0
            self.agents[node] = EvolutionaryAgent(node_id=node, strategy=strategy)
            self.env.update_node_state(node, strategy)

    def _payoff(self, a1, a2):
        if a1 == 1 and a2 == 1: return self.R
        if a1 == 1 and a2 == 0: return self.S
        if a1 == 0 and a2 == 1: return self.T
        return self.P

    def _play_all_games(self):
        """Every agent plays PD with all neighbors. Payoff accumulates."""
        for node in self.agents:
            self.agents[node].reset_round_payoff()
        
        # Play multiple rounds to get a stable payoff signal.
        # This is critical: with only 1 round, a single lucky defector
        # can cascade through the network. With 5+ rounds, the average
        # payoff reflects the TRUE structural advantage of cooperator clusters.
        for _ in range(self.rounds_per_update):
            for node in self.agents:
                neighbors = self.env.get_neighbors(node)
                my_strategy = self.agents[node].strategy
                for neighbor in neighbors:
                    payoff = self._payoff(my_strategy, self.agents[neighbor].strategy)
                    self.agents[node].add_payoff(payoff)

    def _update_best_neighbor(self):
        """
        Nowak & May (1992): Copy the strategy of the highest-earning
        agent in your neighborhood (including yourself). Synchronous.
        
        WHY TOPOLOGY MATTERS (worked example with K=6, T=1.3, R=1.0):
        
        LATTICE (p=0): Cooperators form clusters.
          Interior C: 6 coop neighbors → 6*R*rounds = 6.0*5 = 30.0
          Boundary D: 3C + 3D neighbors → (3*T + 3*P)*5 = 3.9*5 = 19.5
          → Interior C WINS. Cluster holds.
        
        RANDOM (p=1): No clusters. D always finds fresh C to exploit.
          → D quickly dominates.
        """
        new_strategies = {}
        
        for node in self.agents:
            neighbors = list(self.env.get_neighbors(node))
            candidates = [node] + neighbors
            best_node = max(candidates, key=lambda n: self.agents[n].round_payoff)
            new_strategies[node] = self.agents[best_node].strategy
        
        # Apply synchronously
        for node, strategy in new_strategies.items():
            self.agents[node].strategy = strategy
            self.env.update_node_state(node, strategy)

    def _update_fermi(self):
        """Santos & Pacheco (2005): Stochastic pairwise comparison."""
        nodes = list(self.agents.keys())
        random.shuffle(nodes)
        
        for node in nodes:
            neighbors = self.env.get_neighbors(node)
            if not neighbors:
                continue
            
            neighbor = random.choice(neighbors)
            
            my_payoff = self.agents[node].round_payoff
            their_payoff = self.agents[neighbor].round_payoff
            
            max_degree = max(len(self.env.get_neighbors(node)), 
                           len(self.env.get_neighbors(neighbor)), 1)
            D_max = max_degree * max(abs(self.T), abs(self.R), 1.0) * self.rounds_per_update
            
            diff = (their_payoff - my_payoff) / max(D_max, 1.0)
            exponent = max(-50, min(50, -self.beta * diff))
            prob = 1.0 / (1.0 + np.exp(exponent))
            
            if random.random() < prob:
                self.agents[node].strategy = self.agents[neighbor].strategy
                self.env.update_node_state(node, self.agents[neighbor].strategy)

    def step(self):
        """One generation: play multiple rounds → update strategies → maybe mutate."""
        # 1. Play games (multiple rounds for stable payoff)
        self._play_all_games()
        
        # 2. Strategy update
        if self.update_rule == "best_neighbor":
            self._update_best_neighbor()
        else:
            self._update_fermi()
        
        # 3. Mutation: small chance of spontaneous defection.
        # Biased toward defection (not random flip) because we're modeling
        # "temptation invasion" — the interesting question is whether
        # the network can RESIST defection, not random noise.
        if self.mutation_rate > 0:
            for node in self.agents:
                if random.random() < self.mutation_rate:
                    # Flip to the opposite strategy
                    new_strategy = 1 - self.agents[node].strategy
                    self.agents[node].strategy = new_strategy
                    self.env.update_node_state(node, new_strategy)
        
        return self.env.get_cooperation_rate()

    def inject_defectors(self, count):
        """Inject defectors for resilience testing."""
        cooperators = [n for n in self.agents if self.agents[n].strategy == 1]
        count = min(count, len(cooperators))
        targets = random.sample(cooperators, count)
        for node in targets:
            self.agents[node].strategy = 0
            self.env.update_node_state(node, 0)

    def get_strategy_counts(self):
        coop = sum(1 for a in self.agents.values() if a.strategy == 1)
        return coop, len(self.agents) - coop
