"""
Graph Convolutional Multi-Agent RL (GC-MARL) Engine.

Embeds the mathematical adjacency matrix of the network directly into the 
PyTorch Neural Network so agents naturally comprehend their topological structure.
Replaces random epsilon-exploration with Temperature-controlled Softmax (Boltzmann).
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from environment import SocialNetwork
from agent import NeuralAgent
from models import GraphDQN, ReplayBuffer

class SimulationEngine:
    def __init__(self, n=100, k=6, p=0.0,
                 T=1.3, R=1.0, P=0.0, S=0.0,
                 init_coop_fraction=0.8,
                 graph_type="watts_strogatz",
                 learning_rate=1e-3,
                 batch_size=32,
                 gamma=0.95,
                 temperature=1.0,
                 temp_decay=0.995,
                 temp_min=0.05):
        """
        GC-MARL parameters initialized.
        """
        self.env = SocialNetwork(n, k, p, graph_type=graph_type)
        self.n = n
        
        self.T = T
        self.R = R
        self.P = P
        self.S = S
        self.global_step = 0

        self.lr = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.temp = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        
        self.device = torch.device("cpu")
        self.policy_net = GraphDQN().to(self.device)
        self.target_net = GraphDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=10000)
        self.last_loss = 0.0

        # --- Compute Normalized Adjacency Matrix (A_hat) ---
        A = nx.to_numpy_array(self.env.graph)
        # Add self-loops to retain own state
        A_prime = A + np.eye(self.n)
        # Degree matrix (D)
        D = np.sum(A_prime, axis=1)
        D_inv = np.diag(1.0 / D)
        # A_hat = D^-1 * A_prime (normalized aggregation)
        A_hat = np.dot(D_inv, A_prime)
        
        self.env_A_hat = torch.FloatTensor(A_hat).to(self.device)
        self.batch_A_hat = self.env_A_hat.unsqueeze(0).expand(self.batch_size, -1, -1)

        # Create Agents
        self.agents = {}
        all_nodes = list(self.env.graph.nodes())
        num_coop = int(len(all_nodes) * init_coop_fraction)
        coop_nodes = set(random.sample(all_nodes, num_coop))
        
        for node in all_nodes:
            self.agents[node] = NeuralAgent(
                node_id=node, 
                strategy=1 if node in coop_nodes else 0
            )
            self.env.update_node_state(node, self.agents[node].strategy)

    def _payoff(self, a1, a2):
        if a1 == 1 and a2 == 1: return self.R
        if a1 == 1 and a2 == 0: return self.S
        if a1 == 0 and a2 == 1: return self.T
        return self.P

    def _get_node_observations(self):
        """Returns feature matrix X: shape [N, 2] -> [strategy, last_payoff]"""
        X = np.zeros((self.n, 2), dtype=np.float32)
        for i, node in enumerate(self.agents.keys()):
            X[i, 0] = self.agents[node].strategy
            X[i, 1] = self.agents[node].round_payoff
        return X

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(2) # [B, N, 1]
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Q(s, a): forward pass the whole graph batch
        q_values_full = self.policy_net(self.batch_A_hat, states) # [B, N, 2]
        q_values = q_values_full.gather(2, actions).squeeze(2) # [B, N]
        
        # Max Q(s', a') from target net
        with torch.no_grad():
            next_q_values = self.target_net(self.batch_A_hat, next_states).max(2)[0] # [B, N]
            target_q_values = rewards + self.gamma * next_q_values # [B, N]
            
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def step(self):
        """One generation of GC-MARL."""
        self.global_step += 1
        
        # 1. State extraction
        current_state = self._get_node_observations() # [N, 2]
        
        # 2. Select actions via Boltzmann (Softmax) Exploration
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(self.env_A_hat, state_tensor) # [N, 2]
            
            # Scale Q-values by Temperature
            # If temp is high (~1.0), Softmax explores randomly.
            # If temp is low (~0.05), Softmax acts like Argmax (protects clusters).
            scaled_q = q_vals / max(self.temp, 1e-4)
            probs = torch.softmax(scaled_q, dim=1)
            
            # Sample from probability distribution
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

        # 3. Apply actions and reset payoff
        action_history = np.zeros(self.n, dtype=np.int64)
        for i, node in enumerate(self.agents.keys()):
            action = int(actions[i])
            self.agents[node].strategy = action
            action_history[i] = action
            self.env.update_node_state(node, action)
            self.agents[node].reset_round_payoff()
            
        # 4. Play Prisoners Dilemma
        rewards = np.zeros(self.n, dtype=np.float32)
        for i, node in enumerate(self.agents.keys()):
            neighbors = self.env.get_neighbors(node)
            my_strat = self.agents[node].strategy
            for neighbor in neighbors:
                p = self._payoff(my_strat, self.agents[neighbor].strategy)
                self.agents[node].add_payoff(p)
            rewards[i] = self.agents[node].round_payoff
            self.env.update_node_score(node, self.agents[node].round_payoff)
            
        # 5. Get Next State & Store in Replay Buffer
        next_state = self._get_node_observations()
        self.memory.push(current_state, action_history, rewards, next_state)
            
        # 6. Train Network
        loss = self._train_step()
        self.last_loss = loss
        
        # 7. Anneal Temperature
        if self.temp > self.temp_min:
            self.temp *= self.temp_decay
            
        # 8. Update Target Network periodically
        if self.global_step % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return self.env.get_cooperation_rate()

    def get_strategy_counts(self):
        coop = sum(1 for a in self.agents.values() if a.strategy == 1)
        return coop, len(self.agents) - coop

    def inject_defectors(self, count):
        cooperators = [n for n in self.agents if self.agents[n].strategy == 1]
        count = min(count, len(cooperators))
        targets = random.sample(cooperators, count)
        for node in targets:
            self.agents[node].strategy = 0
            self.env.update_node_state(node, 0)
