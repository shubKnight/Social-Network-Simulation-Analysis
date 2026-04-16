"""
Graph Convolutional Multi-Agent RL (GC-MARL) Engine — v3

Key improvements over v2:
  - Heterogeneous agent personalities (Altruist / Grudger / Opportunist / Random)
  - Reward normalization (z-score) to prevent cooperation payoff-scale bias
  - Slower temperature annealing + higher temp_min for adequate exploration
  - Full state vector [strategy, payoff, reputation, personality] fed to GCN
  - Personality-aware reward shaping + optional action overrides
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from collections import Counter

from environment import SocialNetwork
from agent import NeuralAgent
from models import GraphDQN, ReplayBuffer


class SimulationEngine:
    def __init__(self,
                 n=100, k=6, p=0.0,
                 T=1.3, R=1.0, P=0.0, S=0.0,
                 init_coop_fraction=0.5,
                 graph_type="watts_strogatz",
                 learning_rate=1e-3,
                 batch_size=64,
                 gamma=0.95,
                 temperature=2.0,
                 temp_decay=0.99,
                 temp_min=0.2,
                 rewiring_rate=0.3):    # fraction of exploited agents who attempt rewiring
        """
        GC-MARL v3 — personalities enabled, reward normalization active.
        
        init_coop_fraction defaults to 0.5 (50/50 start) for unbiased dynamics.
        temperature defaults to 2.0 (wide early exploration).
        temp_decay defaults to 0.99 (slow annealing — gives network time to train).
        temp_min defaults to 0.1 (never fully deterministic).
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
        self.rewiring_rate = rewiring_rate
        self.last_rewire_count = 0      # rewiring events last step
        self.MIN_DEGREE = 2             # agents must keep at least 2 connections
        self.MAX_DEGREE = max(k * 2, 6) # cap to prevent trivial super-hubs

        self.device = torch.device("cpu")
        self.policy_net = GraphDQN().to(self.device)   # state_dim=4
        self.target_net = GraphDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=20000)
        self.last_loss = 0.0

        # ── Normalized Adjacency Matrix ──────────────────────────────
        A = nx.to_numpy_array(self.env.graph)
        A_prime = A + np.eye(self.n)                   # add self-loops
        D_inv = np.diag(1.0 / A_prime.sum(axis=1))
        A_hat = D_inv @ A_prime                        # row-normalized

        self.env_A_hat = torch.FloatTensor(A_hat).to(self.device)
        self.batch_A_hat = self.env_A_hat.unsqueeze(0).expand(self.batch_size, -1, -1)

        # ── Create Agents with random personalities ───────────────────
        self.agents = {}
        all_nodes = list(self.env.graph.nodes())
        num_coop = int(len(all_nodes) * init_coop_fraction)
        coop_nodes = set(random.sample(all_nodes, num_coop))

        for node in all_nodes:
            self.agents[node] = NeuralAgent(
                node_id=node,
                strategy=1 if node in coop_nodes else 0,
                # personality assigned randomly inside NeuralAgent
            )
            self.env.update_node_state(node, self.agents[node].strategy)

    # ── Helpers ──────────────────────────────────────────────────────

    def _recompute_A_hat(self):
        """
        Recompute the normalized adjacency matrix after topology changes.
        This is the critical step: the GCN now physically sees the new graph
        on the very next forward pass. Rewiring is not cosmetic — it changes
        what information flows through the neural network.
        """
        A = nx.to_numpy_array(self.env.graph)
        A_prime = A + np.eye(self.n)
        D_inv = np.diag(1.0 / A_prime.sum(axis=1))
        A_hat = D_inv @ A_prime
        self.env_A_hat = torch.FloatTensor(A_hat).to(self.device)
        self.batch_A_hat = self.env_A_hat.unsqueeze(0).expand(self.batch_size, -1, -1)

    def _maybe_rewire(self, suckered_nodes: list, defector_this_round: dict):
        """
        Meaningful co-evolutionary rewiring.

        Logic:
          1. Only exploited (suckered) agents consider rewiring.
          2. They sever the tie to the specific neighbor who defected against them
             (not a random one) — if their degree stays above MIN_DEGREE.
          3. They then seek a replacement: scan 2-hop neighbourhood for the node
             with highest reputation who is currently cooperating and not yet
             connected — forming a new trust-based tie.
          4. If no suitable 2-hop candidate exists, skip reconnection (no reward
             for connecting to strangers you know nothing about).
        """
        rewire_count = 0
        rewired_nodes = set()   # avoid double-rewiring same node in one step

        candidates = [n for n in suckered_nodes if random.random() < self.rewiring_rate]
        random.shuffle(candidates)  # random order to avoid systematic bias

        for node in candidates:
            if node in rewired_nodes:
                continue
            ag = self.agents[node]
            neighbors = self.env.get_neighbors(node)
            degree = len(neighbors)

            # ── Step 1: find the worst neighbor (defected against us this round) ──
            betrayers = [nb for nb in neighbors
                         if nb in defector_this_round and defector_this_round[nb] == 0
                         and ag.strategy == 1]   # only betrayers of cooperators
            if not betrayers:
                continue
            # Pick the one with lowest reputation (most chronic defector)
            worst = min(betrayers, key=lambda nb: self.agents[nb].reputation)

            # ── Step 2: check degree constraint ──────────────────────────────────
            if degree <= self.MIN_DEGREE:
                continue   # can't cut — would isolate agent

            # ── Step 3: find a replacement in 2-hop neighbourhood ────────────────
            one_hop   = set(neighbors)
            two_hop   = set()
            for nb in neighbors:
                for nb2 in self.env.get_neighbors(nb):
                    two_hop.add(nb2)
            two_hop -= one_hop
            two_hop.discard(node)

            # Filter: cooperating now, high reputation, degree not saturated
            good_candidates = [
                nb2 for nb2 in two_hop
                if self.agents[nb2].strategy == 1
                and self.agents[nb2].reputation > 0.55
                and len(self.env.get_neighbors(nb2)) < self.MAX_DEGREE
            ]

            if not good_candidates:
                continue   # no worthy candidate found — don't rewire blindly

            # Pick the candidate with highest reputation
            best = max(good_candidates, key=lambda nb2: self.agents[nb2].reputation)

            # ── Step 4: execute rewiring ─────────────────────────────────────────
            self.env.remove_edge(node, worst)
            self.env.add_edge(node, best)
            rewired_nodes.add(node)
            rewire_count += 1

        self.last_rewire_count = rewire_count
        if rewire_count > 0:
            self._recompute_A_hat()   # GCN sees new topology next step

    def _payoff(self, a1, a2):
        if a1 == 1 and a2 == 1: return self.R
        if a1 == 1 and a2 == 0: return self.S
        if a1 == 0 and a2 == 1: return self.T
        return self.P

    def _get_node_observations(self):
        """
        Feature matrix X : [N, 4]
            col 0  strategy            (0 or 1)
            col 1  round_payoff        (raw, unnormalized — GCN learns scale)
            col 2  reputation          (lifetime coop rate 0→1)
            col 3  personality_embed   (-1.0 → +1.0)
        """
        X = np.zeros((self.n, 4), dtype=np.float32)
        for i, node in enumerate(self.agents.keys()):
            ag = self.agents[node]
            X[i, 0] = ag.strategy
            X[i, 1] = ag.round_payoff
            X[i, 2] = ag.reputation
            X[i, 3] = ag.personality_embed
        return X

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Soft normalization: scales rewards to [-1, +1] range while
        preserving the sign and relative ordering.
        Unlike z-score, this keeps the signal that cooperation > defection
        in cooperative neighborhoods, which is essential for the GCN to learn.
        """
        r_min = rewards.min()
        r_max = rewards.max()
        if r_max - r_min < 1e-8:
            # All rewards equal — return small noise to prevent dead gradient
            return np.random.uniform(-0.1, 0.1, size=rewards.shape).astype(np.float32)
        # Scale to [-1, 1]
        return (2.0 * (rewards - r_min) / (r_max - r_min) - 1.0).astype(np.float32)

    # ── Training ─────────────────────────────────────────────────────

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device).unsqueeze(2)   # [B, N, 1]
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Q(s, a)
        q_full  = self.policy_net(self.batch_A_hat, states)          # [B, N, 2]
        q_taken = q_full.gather(2, actions).squeeze(2)               # [B, N]

        # Bellman target
        with torch.no_grad():
            next_q   = self.target_net(self.batch_A_hat, next_states).max(2)[0]  # [B, N]
            q_target = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # ── Main step ────────────────────────────────────────────────────

    def step(self):
        """One generation of GC-MARL with personality-aware dynamics."""
        self.global_step += 1

        # 1. Observe current state
        current_state = self._get_node_observations()   # [N, 4]

        # 2. Select actions via Boltzmann exploration
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        with torch.no_grad():
            q_vals    = self.policy_net(self.env_A_hat, state_tensor)  # [N, 2]
            scaled_q  = q_vals / max(self.temp, 1e-4)
            probs     = torch.softmax(scaled_q, dim=1)
            dqn_actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

        # 3. Apply personality overrides + update agent state
        action_history = np.zeros(self.n, dtype=np.int64)
        nodes = list(self.agents.keys())
        for i, node in enumerate(nodes):
            ag = self.agents[node]
            final_action = ag.maybe_override_action(int(dqn_actions[i]))
            ag.strategy = final_action
            ag.update_reputation(final_action)
            action_history[i] = final_action
            self.env.update_node_state(node, final_action)
            ag.reset_round_payoff()

        # 4. Play Prisoner's Dilemma
        raw_rewards = np.zeros(self.n, dtype=np.float32)
        for i, node in enumerate(nodes):
            ag = self.agents[node]
            neighbors = self.env.get_neighbors(node)
            coop_neighbor_count = 0
            for neighbor in neighbors:
                opp_action = self.agents[neighbor].strategy
                p = self._payoff(ag.strategy, opp_action)
                ag.add_payoff(p, opponent_action=opp_action)
                if opp_action == 1:
                    coop_neighbor_count += 1
            raw_rewards[i] = ag.round_payoff
            self.env.update_node_score(node, ag.round_payoff)

            coop_ratio = coop_neighbor_count / max(len(neighbors), 1)
            ag.update_grudger_state(coop_ratio)

            # 4b. Personality reward shaping
            raw_rewards[i] = ag.personality_adjust_reward(
                raw_rewards[i],
                coop_neighbor_count,
                len(neighbors)
            )

        # 4c. Dynamic rewiring — suckered agents try to cut defectors and
        #     reconnect to reputable cooperators in their 2-hop neighbourhood.
        #     Runs after payoffs so we know who was suckered this round.
        suckered = [node for node in nodes if self.agents[node].was_suckered]
        action_map = {node: self.agents[node].strategy for node in nodes}
        self._maybe_rewire(suckered, action_map)

        # 5. Normalize rewards to prevent payoff-scale bias
        norm_rewards = self._normalize_rewards(raw_rewards)

        # 6. Observe next state & store experience
        next_state = self._get_node_observations()
        self.memory.push(current_state, action_history, norm_rewards, next_state)

        # 7. Train GCN
        self.last_loss = self._train_step()

        # 8. Anneal temperature
        if self.temp > self.temp_min:
            self.temp = max(self.temp_min, self.temp * self.temp_decay)

        # 9. Sync target network
        if self.global_step % 20 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.env.get_cooperation_rate()

    # ── Utility ──────────────────────────────────────────────────────

    def get_strategy_counts(self):
        coop = sum(1 for a in self.agents.values() if a.strategy == 1)
        return coop, len(self.agents) - coop

    def get_personality_counts(self):
        """Returns dict of personality -> count for dashboard display."""
        counts = Counter(a.personality for a in self.agents.values())
        return dict(counts)

    def get_personality_coop_rates(self):
        """Returns dict of personality -> current cooperation rate."""
        stats = {}
        for ptype in ["altruist", "grudger", "opportunist", "random"]:
            agents_of_type = [a for a in self.agents.values() if a.personality == ptype]
            if agents_of_type:
                rate = sum(a.strategy for a in agents_of_type) / len(agents_of_type)
                stats[ptype] = rate
            else:
                stats[ptype] = 0.0
        return stats

    def inject_defectors(self, count):
        """Shock: forcibly convert cooperators to defectors."""
        cooperators = [n for n in self.agents if self.agents[n].strategy == 1]
        count = min(count, len(cooperators))
        for node in random.sample(cooperators, count):
            self.agents[node].strategy = 0
            self.env.update_node_state(node, 0)
