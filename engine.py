"""
Graph Convolutional Multi-Agent RL (GC-MARL) Engine — v4

What changed from v3:
  - Removed all hardcoded personality types and reward shaping.
  - Agents now build behavioral profiles from their own experience:
      strategy_trend  = rolling mean of last 20 actions
      payoff_trend    = rolling normalized payoff
      betrayal_rate   = fraction of cooperative moves that were exploited
  - State vector: [strategy, payoff, reputation, strategy_trend,
                   payoff_trend, betrayal_rate]  (dim=6)
  - The GCN learns differentiated strategies purely through gradient descent.
  - Dynamic network rewiring retained: suckered agents cut chronic defectors
    and seek reputable cooperators in 2-hop neighbourhood.
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
    def __init__(self,
                 n=100, k=6, p=0.0,
                 T=1.3, R=1.0, P=0.0, S=0.0,
                 init_coop_fraction=0.5,
                 graph_type="watts_strogatz",
                 learning_rate=1e-3,
                 batch_size=64,
                 gamma=0.99,
                 temperature=2.0,
                 temp_decay=0.995,
                 temp_min=0.2,
                 temp_warmup=100,       # steps before temperature starts decaying
                 rewiring_rate=0.3):
        self.env = SocialNetwork(n, k, p, graph_type=graph_type)
        self.n   = n
        self.T   = T
        self.R   = R
        self.P   = P
        self.S   = S
        self.global_step = 0

        self.lr          = learning_rate
        self.batch_size  = batch_size
        self.gamma       = gamma
        self.temp        = temperature
        self.temp_min    = temp_min
        self.temp_decay  = temp_decay
        self.temp_warmup = temp_warmup
        self.rewiring_rate     = rewiring_rate
        self.last_rewire_count = 0
        self.MIN_DEGREE  = 2
        self.MAX_DEGREE  = max(k * 2, 6)

        self.device      = torch.device("cpu")
        self.policy_net  = GraphDQN().to(self.device)   # state_dim=6
        self.target_net  = GraphDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory      = ReplayBuffer(capacity=20000)
        self.last_loss   = 0.0

        # ── Normalized Adjacency Matrix ──────────────────────────────
        self._recompute_A_hat()

        # ── Create Agents ────────────────────────────────────────────
        # Estimate max possible payoff for payoff_trend normalization
        max_payoff = T * k           # worst case: full neighbour temptation

        self.agents  = {}
        all_nodes    = list(self.env.graph.nodes())
        num_coop     = int(len(all_nodes) * init_coop_fraction)
        coop_nodes   = set(random.sample(all_nodes, num_coop))

        for node in all_nodes:
            self.agents[node] = NeuralAgent(
                node_id    = node,
                strategy   = 1 if node in coop_nodes else 0,
                max_payoff = max_payoff,
            )
            self.env.update_node_state(node, self.agents[node].strategy)

    # ── Graph Utilities ──────────────────────────────────────────────

    def _recompute_A_hat(self):
        """Recompute normalized adjacency matrix. Called at init and after rewiring."""
        A       = nx.to_numpy_array(self.env.graph)
        A_prime = A + np.eye(self.n)
        D_inv   = np.diag(1.0 / A_prime.sum(axis=1))
        A_hat   = D_inv @ A_prime
        self.env_A_hat   = torch.FloatTensor(A_hat).to(self.device)
        self.batch_A_hat = self.env_A_hat.unsqueeze(0).expand(self.batch_size, -1, -1)

    # ── State ────────────────────────────────────────────────────────

    def _get_node_observations(self):
        """
        Feature matrix X : [N, 6]
          col 0  strategy        current action (0/1)
          col 1  round_payoff    raw this-round payoff
          col 2  reputation      lifetime cooperation rate
          col 3  strategy_trend  rolling mean of last 20 actions
          col 4  payoff_trend    rolling normalized payoff [-1, 1]
          col 5  betrayal_rate   fraction of coop moves that were exploited
        All features emerge from experience — no pre-assignment.
        """
        X = np.zeros((self.n, 6), dtype=np.float32)
        for i, node in enumerate(self.agents.keys()):
            ag       = self.agents[node]
            X[i, 0] = ag.strategy
            X[i, 1] = ag.round_payoff
            X[i, 2] = ag.reputation
            X[i, 3] = ag.strategy_trend
            X[i, 4] = ag.payoff_trend
            X[i, 5] = ag.betrayal_rate
        return X

    # ── Reward Normalization ─────────────────────────────────────────

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Scale to [-1, +1] while preserving relative ordering.
        Cooperation's natural payoff advantage is preserved (unlike z-score).
        """
        r_min = rewards.min()
        r_max = rewards.max()
        if r_max - r_min < 1e-8:
            return np.random.uniform(-0.1, 0.1, size=rewards.shape).astype(np.float32)
        return (2.0 * (rewards - r_min) / (r_max - r_min) - 1.0).astype(np.float32)

    # ── Payoff ───────────────────────────────────────────────────────

    def _payoff(self, a1, a2):
        if a1 == 1 and a2 == 1: return self.R
        if a1 == 1 and a2 == 0: return self.S
        if a1 == 0 and a2 == 1: return self.T
        return self.P

    # ── Training ─────────────────────────────────────────────────────

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device).unsqueeze(2)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)

        q_full  = self.policy_net(self.batch_A_hat, states)
        q_taken = q_full.gather(2, actions).squeeze(2)

        with torch.no_grad():
            next_q   = self.target_net(self.batch_A_hat, next_states).max(2)[0]
            q_target = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # ── Rewiring ─────────────────────────────────────────────────────

    def _maybe_rewire(self, suckered_nodes, action_map):
        """
        Co-evolutionary rewiring:
          - Only suckered agents consider rewiring.
          - Cut the specific betrayer with lowest reputation.
          - Seek replacement in 2-hop with highest reputation + cooperating.
          - Recompute A_hat so GCN sees new topology next step.
        """
        rewire_count  = 0
        rewired       = set()

        # Expand beyond suckered: any cooperator whose worst neighbor is a
        # chronic defector (betrayal_rate > 0.5) can consider rewiring.
        # This keeps selection pressure on defectors even after suckering stops.
        all_coops = [n for n in self.agents
                     if self.agents[n].strategy == 1
                     and any(action_map.get(nb) == 0 and
                             self.agents[nb].betrayal_rate > 0.5
                             for nb in self.env.get_neighbors(n))]
        candidates = list(set(suckered_nodes) | set(all_coops))
        candidates = [n for n in candidates if random.random() < self.rewiring_rate]
        random.shuffle(candidates)

        for node in candidates:
            if node in rewired:
                continue
            neighbors = self.env.get_neighbors(node)
            if len(neighbors) <= self.MIN_DEGREE:
                continue

            betrayers = [nb for nb in neighbors
                         if action_map.get(nb) == 0
                         and self.agents[node].strategy == 1]
            if not betrayers:
                continue
            worst = min(betrayers, key=lambda nb: self.agents[nb].reputation)

            # 2-hop search for a trustworthy cooperator
            one_hop = set(neighbors)
            two_hop = set()
            for nb in neighbors:
                for nb2 in self.env.get_neighbors(nb):
                    two_hop.add(nb2)
            two_hop -= one_hop
            two_hop.discard(node)

            good = [nb2 for nb2 in two_hop
                    if self.agents[nb2].strategy == 1
                    and self.agents[nb2].reputation > 0.4
                    and len(self.env.get_neighbors(nb2)) <= self.MAX_DEGREE]
            if not good:
                continue

            best = max(good, key=lambda nb2: self.agents[nb2].reputation)
            self.env.remove_edge(node, worst)
            self.env.add_edge(node, best)
            rewired.add(node)
            rewire_count += 1

        self.last_rewire_count = rewire_count
        if rewire_count > 0:
            self._recompute_A_hat()

    # ── Main Step ────────────────────────────────────────────────────

    def step(self):
        """One generation of GC-MARL with emergent behavioral specialization."""
        self.global_step += 1

        # 1. Observe state
        current_state = self._get_node_observations()

        # 2. Boltzmann action selection
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        with torch.no_grad():
            q_vals  = self.policy_net(self.env_A_hat, state_tensor)
            scaled  = q_vals / max(self.temp, 1e-4)
            probs   = torch.softmax(scaled, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

        # 3. Apply actions
        nodes          = list(self.agents.keys())
        action_history = np.zeros(self.n, dtype=np.int64)
        for i, node in enumerate(nodes):
            ag            = self.agents[node]
            action        = int(actions[i])
            ag.strategy   = action
            action_history[i] = action
            self.env.update_node_state(node, action)
            ag.reset_round_payoff()

        # 4. Play Prisoner's Dilemma — payoffs normalized by degree
        # Isolated defectors earn LESS because they have fewer victims to exploit
        raw_rewards = np.zeros(self.n, dtype=np.float32)
        for i, node in enumerate(nodes):
            ag        = self.agents[node]
            neighbors = self.env.get_neighbors(node)
            n_neighbors = max(len(neighbors), 1)
            for neighbor in neighbors:
                opp = self.agents[neighbor].strategy
                ag.add_payoff(self._payoff(ag.strategy, opp), opponent_action=opp)
            # Normalize by neighbour count: payoff per connection, not total
            # This means defectors who lose connections via rewiring earn less
            raw_rewards[i] = ag.round_payoff / n_neighbors
            self.env.update_node_score(node, ag.round_payoff)

        # 5. Update each agent's behavioral history (emergent "personality")
        for i, node in enumerate(nodes):
            self.agents[node].update_history(action_history[i])

        # 6. Rewiring
        suckered   = [nd for nd in nodes if self.agents[nd].was_suckered]
        action_map = {nd: self.agents[nd].strategy for nd in nodes}
        self._maybe_rewire(suckered, action_map)

        # 7. Normalize & store experience
        norm_rewards = self._normalize_rewards(raw_rewards)
        next_state   = self._get_node_observations()
        self.memory.push(current_state, action_history, norm_rewards, next_state)

        # 8. Train GCN
        self.last_loss = self._train_step()

        # 9. Anneal temperature (only after warmup period)
        if self.global_step > self.temp_warmup and self.temp > self.temp_min:
            self.temp = max(self.temp_min, self.temp * self.temp_decay)

        # 10. Sync target network
        if self.global_step % 20 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.env.get_cooperation_rate()

    # ── Analytics ────────────────────────────────────────────────────

    def get_strategy_counts(self):
        coop = sum(1 for a in self.agents.values() if a.strategy == 1)
        return coop, len(self.agents) - coop

    def get_behavioral_profile(self):
        """
        Returns aggregated emergent behavioral stats across all agents.
        Shows what 'personality distribution' emerged without pre-assignment.
        """
        trends   = [a.strategy_trend  for a in self.agents.values()]
        betrayal = [a.betrayal_rate   for a in self.agents.values()]
        payoffs  = [a.payoff_trend    for a in self.agents.values()]

        # Classify emergent archetypes purely from behavior
        chronic_coops  = sum(1 for t in trends if t > 0.7)
        chronic_defs   = sum(1 for t in trends if t < 0.3)
        high_betrayed  = sum(1 for b in betrayal if b > 0.4)
        swing_agents   = len(trends) - chronic_coops - chronic_defs

        return {
            "avg_strategy_trend":  np.mean(trends),
            "avg_betrayal_rate":   np.mean(betrayal),
            "avg_payoff_trend":    np.mean(payoffs),
            "chronic_cooperators": chronic_coops,
            "chronic_defectors":   chronic_defs,
            "swing_agents":        swing_agents,
            "high_betrayal":       high_betrayed,
        }

    def inject_defectors(self, count):
        cooperators = [n for n in self.agents if self.agents[n].strategy == 1]
        count = min(count, len(cooperators))
        for node in random.sample(cooperators, count):
            self.agents[node].strategy = 0
            self.env.update_node_state(node, 0)
