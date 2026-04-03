from environment import SocialNetwork
from agent import RLAgent
import numpy as np

class SimulationEngine:
    def __init__(self, n=100, k=4, p=0.1, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the simulation engine with a graph and RL agents.
        """
        self.env = SocialNetwork(n, k, p)
        self.agents = {}
        
        # Initialize agents
        for node in self.env.graph.nodes():
            self.agents[node] = RLAgent(node_id=node, alpha=alpha, gamma=gamma, epsilon=epsilon)
            
        # Payoff Matrix (T, R, P, S)
        self.T = 5 # Temptation to defect
        self.R = 3 # Reward for mutual cooperation
        self.P = 1 # Punishment for mutual defection
        self.S = 0 # Sucker's payoff

    def get_coop_percentage(self, node_id):
        """Calculates the percentage of cooperating neighbors for a given node."""
        neighbors = self.env.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        coops = sum(1 for n in neighbors if self.env.get_node_state(n) == 1)
        return coops / len(neighbors)

    def calculate_payoff(self, action1, action2):
        """Returns the payoff for agent 1 playing action1 against agent 2 playing action2."""
        if action1 == 1 and action2 == 1:
            return self.R
        elif action1 == 1 and action2 == 0:
            return self.S
        elif action1 == 0 and action2 == 1:
            return self.T
        else: # both defect
            return self.P

    def step(self):
        """Performs one synchronous round of the game for all agents."""
        actions = {}
        
        # 1. Observation and Action Selection based on current visible state
        for node in self.agents:
            coop_pct = self.get_coop_percentage(node)
            actions[node] = self.agents[node].choose_action(coop_pct)
            
        # 2. Update states synchronously
        for node, action in actions.items():
            self.env.update_node_state(node, action)
            
        # 3. Payoff Calculation
        rewards = {node: 0.0 for node in self.agents}
        
        for node in self.agents:
            neighbors = self.env.get_neighbors(node)
            node_action = actions[node]
            
            node_reward = 0
            for neighbor in neighbors:
                neighbor_action = actions[neighbor]
                node_reward += self.calculate_payoff(node_action, neighbor_action)
                
            rewards[node] = node_reward
            self.env.update_node_score(node, node_reward)
            
        # 4. Learning step (Bellman update) using the *new* state resulting from everyone's actions
        for node in self.agents:
            next_coop_pct = self.get_coop_percentage(node)
            self.agents[node].learn(next_coop_pct, rewards[node])
            
        # Decay epsilon slightly to favor exploitation over time (optional, simplified here)
        # We can implement an explicit decay later if necessary, for now agents keep exploring.
            
        return self.env.get_cooperation_rate(), rewards
