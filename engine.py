"""
Graph Convolutional Multi-Agent RL (GC-MARL) Engine — v5

What changed from v4:
  - Replaced single-integer trait with full OCEAN 5-dimensional personality model.
  - Each agent has: openness, agreeableness, conscientiousness, extraversion, neuroticism.
  - State vector expanded to 11 dimensions:
      [strategy, payoff, reputation, strategy_trend, payoff_trend, betrayal_rate,
       openness, agreeableness, conscientiousness, extraversion, neuroticism]
  - Per-agent Boltzmann temperature (scaled by openness).
  - Conscientiousness-modulated Q-value sharpness.
  - Agreeableness gates rewiring probability and trust rebuilding.
  - Extraversion controls rewiring search radius and max preferred degree.
  - Neuroticism amplifies recent negative payoff experiences.
  - Trait-based homophily replaced with personality-distance homophily.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from environment import SocialNetwork
from agent import NeuralAgent, OCEAN_DIMS
from models import GraphDQN, ReplayBuffer

# Personality-distance threshold for homophily rewiring.
# Agents prefer connecting to others within this distance.
# With normal(0.5, 0.2) in 5D, average pairwise distance ≈ 0.28.
# 0.30 balances cluster formation with network stability.
HOMOPHILY_THRESHOLD = 0.30


class SimulationEngine:
    def __init__(self,
                 n=100, k=6, p=0.0,
                 T=1.5, R=1.0, P=0.1, S=-0.3,
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
        self.MAX_DEGREE  = max(k * 2, 6)    # hard cap (matches original engine)

        self.device      = torch.device("cpu")
        self.policy_net  = GraphDQN(state_dim=11, hidden_dim=128).to(self.device)
        self.target_net  = GraphDQN(state_dim=11, hidden_dim=128).to(self.device)
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

        for node in all_nodes:
            ag = NeuralAgent(
                node_id    = node,
                strategy   = 1,       # temporary, overridden below
                max_payoff = max_payoff,
            )
            # Initial strategy follows personality:
            # positive propensity → cooperate, negative → defect,
            # with some randomness for agents near zero.
            if ag.cooperation_propensity > 0.05:
                ag.strategy = 1
            elif ag.cooperation_propensity < -0.05:
                ag.strategy = 0
            else:
                ag.strategy = random.choice([0, 1])
            self.agents[node] = ag
            self.env.update_node_state(node, ag.strategy)

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
        Feature matrix X : [N, 11]
          col 0   strategy        current action (0/1)
          col 1   round_payoff    raw this-round payoff
          col 2   reputation      lifetime cooperation rate
          col 3   strategy_trend  rolling mean of last 20 actions
          col 4   payoff_trend    rolling normalized payoff [-1, 1]
          col 5   betrayal_rate   fraction of coop moves that were exploited
          col 6   openness        OCEAN personality dimension
          col 7   agreeableness   OCEAN personality dimension
          col 8   conscientiousness OCEAN personality dimension
          col 9   extraversion    OCEAN personality dimension
          col 10  neuroticism     OCEAN personality dimension

        First 6 features emerge from experience; last 5 are intrinsic traits.
        """
        X = np.zeros((self.n, 11), dtype=np.float32)
        for i, node in enumerate(self.agents.keys()):
            ag       = self.agents[node]
            # Behavioral features (emergent)
            X[i, 0] = ag.strategy
            X[i, 1] = ag.round_payoff
            X[i, 2] = ag.reputation
            X[i, 3] = ag.strategy_trend
            X[i, 4] = ag.weighted_payoff_trend   # neuroticism-weighted (not raw)
            X[i, 5] = ag.betrayal_rate
            # Personality features (intrinsic OCEAN)
            X[i, 6]  = ag.openness
            X[i, 7]  = ag.agreeableness
            X[i, 8]  = ag.conscientiousness
            X[i, 9]  = ag.extraversion
            X[i, 10] = ag.neuroticism
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

        # Huber Loss (SmoothL1Loss) handles massive spikes much better than MSELoss
        loss = nn.SmoothL1Loss()(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # ── Rewiring ─────────────────────────────────────────────────────

    def _maybe_rewire(self, suckered_nodes, action_map):
        """
        Co-evolutionary rewiring with personality-driven homophily:
          - Only suckered agents or cooperators near chronic defectors consider rewiring.
          - Agreeableness gates rewiring probability (forgiving agents rewire less).
          - Cut the specific betrayer with lowest reputation.
          - Seek replacement via personality-distance homophily (similar personalities attract).
          - Extraversion controls search radius (high-E searches globally).
          - Recompute A_hat so GCN sees new topology next step.
        """
        rewire_count  = 0
        rewired       = set()

        # Expand beyond suckered: any cooperator whose worst neighbor is a
        # chronic defector (betrayal_rate > threshold) can consider rewiring.
        all_coops = [n for n in self.agents
                     if self.agents[n].strategy == 1
                     and any(action_map.get(nb) == 0 and
                             self.agents[nb].reputation < 0.4
                             for nb in self.env.get_neighbors(n))]

        # Personality-distance homophily pruning: agents connected to very
        # personality-dissimilar neighbors consider rewiring those ties.
        homophily_candidates = [
            n for n in self.agents
            if any(self.agents[n].personality_distance(self.agents[nb]) > HOMOPHILY_THRESHOLD
                   for nb in self.env.get_neighbors(n))
        ]

        candidates = list(set(suckered_nodes) | set(all_coops) | set(homophily_candidates))

        # Use the base rewiring_rate directly — personality modulation happens
        # in who gets cut and who gets chosen, not in whether rewiring fires.
        candidates = [n for n in candidates
                      if random.random() < self.rewiring_rate]
        random.shuffle(candidates)

        for node in candidates:
            if node in rewired:
                continue
            neighbors = self.env.get_neighbors(node)
            if len(neighbors) <= self.MIN_DEGREE:
                continue

            ag = self.agents[node]

            # Prioritize cutting defectors. If no defectors, cut personality-dissimilar.
            betrayers = [nb for nb in neighbors if action_map.get(nb) == 0]
            if betrayers:
                worst = min(betrayers, key=lambda nb: self.agents[nb].reputation)
            else:
                # Cut the most personality-dissimilar neighbor
                dissimilar = sorted(neighbors,
                                    key=lambda nb: ag.personality_distance(self.agents[nb]),
                                    reverse=True)
                if dissimilar and ag.personality_distance(self.agents[dissimilar[0]]) > HOMOPHILY_THRESHOLD:
                    worst = dissimilar[0]
                else:
                    continue

            # Hard MAX_DEGREE cap, but extraversion softens it slightly
            max_deg = min(self.MAX_DEGREE, ag.max_preferred_degree)

            # 2-hop search for a personality-similar agent (homophily-driven).
            # NO hard strategy filter — personality similarity is the primary
            # driver. Agreeable agents will find agreeable agents (who tend to
            # cooperate), neurotic agents find neurotic agents (who tend to
            # defect). This creates personality-based clusters, not
            # strategy-based echo chambers.
            one_hop = set(neighbors)
            two_hop = set()
            for nb in neighbors:
                for nb2 in self.env.get_neighbors(nb):
                    two_hop.add(nb2)
            two_hop -= one_hop
            two_hop.discard(node)

            good = [nb2 for nb2 in two_hop
                    if ag.personality_distance(self.agents[nb2]) < HOMOPHILY_THRESHOLD
                    and len(self.env.get_neighbors(nb2)) <= max_deg]

            if not good:
                # High-extraversion agents search globally when 2-hop fails
                if ag.extraversion > 0.5:
                    good = [w for w in range(self.n)
                            if w != node and not self.env.graph.has_edge(node, w)
                            and ag.personality_distance(self.agents[w]) < HOMOPHILY_THRESHOLD
                            and len(self.env.get_neighbors(w)) <= max_deg]

                if not good:
                    # Fallback: relax homophily, find any non-neighbor
                    good = [nb2 for nb2 in two_hop
                            if len(self.env.get_neighbors(nb2)) <= max_deg]

                if not good:
                    continue

            # Pick the most personality-similar among valid candidates,
            # with reputation as a soft tiebreaker (not a hard filter)
            best = max(good, key=lambda nb2: (
                ag.personality_similarity(self.agents[nb2]) * 0.7 +
                self.agents[nb2].reputation * 0.3
            ))

            self.env.remove_edge(node, worst)
            self.env.add_edge(node, best)

            if self.env.graph.has_edge(node, best):
                # Smart labeling: personality-similar matches are 'local' (kindred spirit),
                # only true strangers from the fallback path get 'random'.
                dist = ag.personality_distance(self.agents[best])
                if dist < HOMOPHILY_THRESHOLD:
                    self.env.graph.edges[node, best]['edge_type'] = 'local'
                    self.env.graph.edges[node, best]['edge_trust'] = 0.5
                else:
                    self.env.graph.edges[node, best]['edge_type'] = 'random'
                    self.env.graph.edges[node, best]['edge_trust'] = 0.1
            rewired.add(node)
            rewire_count += 1

            # Cap rewires per step to prevent full-graph churn
            max_rewires = max(3, int(self.env.graph.number_of_edges() * 0.15))
            if rewire_count >= max_rewires:
                break

        self.last_rewire_count = rewire_count
        if rewire_count > 0:
            self._recompute_A_hat()

    # ── Main Step ────────────────────────────────────────────────────

    def step(self):
        """One generation of GC-MARL with OCEAN personality modulation."""
        self.global_step += 1

        # 1. Observe state (11-dim per node)
        current_state = self._get_node_observations()

        # 2. Per-agent Boltzmann action selection with personality modulation
        state_tensor = torch.FloatTensor(current_state).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(self.env_A_hat, state_tensor)   # [N, 2]

            # Apply conscientiousness-based Q-sharpness per agent
            nodes = list(self.agents.keys())
            sharpness = torch.FloatTensor([
                self.agents[n].q_sharpness for n in nodes
            ]).unsqueeze(1).to(self.device)   # [N, 1]
            q_vals = q_vals * sharpness

            # === DIRECT PERSONALITY BIAS (the key mechanism) ===
            # Add cooperation propensity as an additive bias to Q-values.
            # q_vals[:, 0] = defect, q_vals[:, 1] = cooperate
            # Positive propensity → boost cooperate, penalize defect.
            propensity = torch.FloatTensor([
                self.agents[n].cooperation_propensity for n in nodes
            ]).to(self.device)   # [N]
            q_vals[:, 1] += propensity    # cooperate gets +bias
            q_vals[:, 0] -= propensity    # defect gets -bias

            # Per-agent temperature: base temp × openness scaling
            per_agent_temp = torch.FloatTensor([
                max(self.agents[n].effective_temperature(self.temp), 1e-4)
                for n in nodes
            ]).unsqueeze(1).to(self.device)   # [N, 1]

            scaled  = q_vals / per_agent_temp
            probs   = torch.softmax(scaled, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

        # 3. Apply actions (with trauma lockout override)
        action_history = np.zeros(self.n, dtype=np.int64)
        for i, node in enumerate(nodes):
            ag            = self.agents[node]
            action        = int(actions[i])

            # Trauma lockout: forced defection, GCN cannot override
            if ag.trauma_lockout > 0:
                action = 0
                ag.trauma_lockout -= 1

            ag.strategy   = action
            action_history[i] = action
            self.env.update_node_state(node, action)
            ag.reset_round_payoff()

        # 4. Play Prisoner's Dilemma — payoffs normalized by degree
        # Payoffs are scaled by dynamic `edge_trust`. Mutual cooperation builds trust.
        raw_rewards = np.zeros(self.n, dtype=np.float32)
        edge_trust_updates = {}
        for i, node in enumerate(nodes):
            ag        = self.agents[node]
            neighbors = self.env.get_neighbors(node)
            n_neighbors = max(len(neighbors), 1)

            # Dictionary to track round events for dynamic personality drift
            round_events = {
                'mutual_coop': 0,
                'suckered': 0,
                'mutual_defection': 0,
                'stranger_success': False,
                'stranger_betrayal': False,
                'avg_payoff': 0.0,
                'degree': len(neighbors)
            }

            for neighbor in neighbors:
                opp = self.agents[neighbor].strategy
                base_payoff = self._payoff(ag.strategy, opp)

                # Fetch dynamic edge trust (starts at 1.0 for local, 0.1 for random)
                edge_data = self.env.graph.edges.get((node, neighbor), {})
                edge_trust = edge_data.get('edge_trust', 1.0)
                edge_type  = edge_data.get('edge_type', 'local')

                # Log events for drift
                if ag.strategy == 1 and opp == 1:
                    round_events['mutual_coop'] += 1
                elif ag.strategy == 1 and opp == 0:
                    round_events['suckered'] += 1
                elif ag.strategy == 0 and opp == 0:
                    round_events['mutual_defection'] += 1
                
                if edge_type == 'random':
                    if base_payoff > 0 and opp == 1:
                        round_events['stranger_success'] = True
                    elif opp == 0:
                        round_events['stranger_betrayal'] = True

                # --- Subjective Utility Modifiers ---
                # Moderate strength: enough to differentiate personalities in RL
                # but not enough to make R > T (which kills defection entirely).

                # 1. Agreeableness: warm-glow for cooperation, guilt for exploitation
                if ag.strategy == 1 and opp == 1:
                    base_payoff += (ag.agreeableness - 0.5) * 0.6    # up to ±0.3
                elif ag.strategy == 0 and opp == 1:
                    base_payoff -= (ag.agreeableness - 0.5) * 0.6    # guilt penalty

                # 2. Neuroticism: betrayal panic, conflict dread
                if ag.strategy == 1 and opp == 0:
                    base_payoff -= (ag.neuroticism - 0.5) * 0.5      # suckered pain
                elif ag.strategy == 0 and opp == 0:
                    base_payoff -= (ag.neuroticism - 0.5) * 0.4      # conflict stress

                # 3. Conscientiousness: values established trust
                if edge_trust > 0.8:
                    base_payoff += (ag.conscientiousness - 0.5) * 0.4

                # 4. Openness: thrill of novel connections
                if edge_type == 'random':
                    base_payoff += (ag.openness - 0.5) * 0.5

                # Scale subjective payoff by trust: you earn little from strangers until trust settles
                ag.add_payoff(base_payoff * edge_trust, opponent_action=opp)

                # Calculate trust evolution (only calculate once per undirected edge)
                edge_id = tuple(sorted((node, neighbor)))
                if edge_id not in edge_trust_updates:
                    # Trust rebuild speed modulated by the PAIR's average agreeableness
                    pair_agree = (ag.agreeableness +
                                  self.agents[neighbor].agreeableness) / 2.0
                    if ag.strategy == 1 and opp == 1:
                        # Mutual cooperation builds trust. High agreeableness → faster.
                        trust_gain = 0.05 + 0.15 * pair_agree   # [0.05 .. 0.20]
                        edge_trust_updates[edge_id] = min(1.0, edge_trust + trust_gain)
                    elif ag.strategy == 0 or opp == 0:
                        # Defection shatters trust. Low agreeableness → more severe.
                        trust_loss = 0.3 + 0.3 * (1.0 - pair_agree)   # [0.3 .. 0.6]
                        edge_trust_updates[edge_id] = max(0.0, edge_trust - trust_loss)

            # --- 4. Extraversion gives a macro-level utility for high degree ---
            social_bonus = (ag.extraversion - 0.5) * n_neighbors * 0.05
            ag.add_payoff(social_bonus)

            # Normalize by neighbour count for RL
            raw_rewards[i] = ag.round_payoff / n_neighbors
            self.env.update_node_score(node, ag.round_payoff)
            
            # --- Dynamic Personality Drift ---
            # Now that the round is complete for this agent, shift traits slightly
            round_events['avg_payoff'] = raw_rewards[i]
            ag.apply_drift(round_events)

        # Apply the edge trust changes back to the graph synchronously
        for (u, v), new_trust in edge_trust_updates.items():
            self.env.update_edge_trust(u, v, new_trust)

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
        Returns aggregated emergent behavioral stats across all agents,
        including personality distribution breakdown.
        """
        trends   = [a.strategy_trend  for a in self.agents.values()]
        betrayal = [a.betrayal_rate   for a in self.agents.values()]
        payoffs  = [a.payoff_trend    for a in self.agents.values()]

        # Classify emergent archetypes purely from behavior
        chronic_coops  = sum(1 for t in trends if t > 0.7)
        chronic_defs   = sum(1 for t in trends if t < 0.3)
        high_betrayed  = sum(1 for b in betrayal if b > 0.4)
        swing_agents   = len(trends) - chronic_coops - chronic_defs

        # Personality statistics
        personality_stats = {}
        for dim in OCEAN_DIMS:
            vals = [a.personality[dim] for a in self.agents.values()]
            personality_stats[dim] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
                'min':  float(np.min(vals)),
                'max':  float(np.max(vals)),
            }

        # Cooperation rate by personality dimension (binned into Low/Mid/High)
        coop_by_personality = {}
        for dim in OCEAN_DIMS:
            low_coop  = [a.strategy_trend for a in self.agents.values()
                         if a.personality[dim] < 0.33]
            mid_coop  = [a.strategy_trend for a in self.agents.values()
                         if 0.33 <= a.personality[dim] <= 0.67]
            high_coop = [a.strategy_trend for a in self.agents.values()
                         if a.personality[dim] > 0.67]
            coop_by_personality[dim] = {
                'low':  float(np.mean(low_coop))  if low_coop  else 0.0,
                'mid':  float(np.mean(mid_coop))  if mid_coop  else 0.0,
                'high': float(np.mean(high_coop)) if high_coop else 0.0,
            }

        return {
            "avg_strategy_trend":  np.mean(trends),
            "avg_betrayal_rate":   np.mean(betrayal),
            "avg_payoff_trend":    np.mean(payoffs),
            "chronic_cooperators": chronic_coops,
            "chronic_defectors":   chronic_defs,
            "swing_agents":        swing_agents,
            "high_betrayal":       high_betrayed,
            "personality_stats":   personality_stats,
            "coop_by_personality": coop_by_personality,
        }

    def get_personality_data(self):
        """
        Returns per-agent personality data for visualization.
        List of dicts with node_id, OCEAN values, strategy, cooperation_trend.
        """
        data = []
        for node, ag in self.agents.items():
            entry = {
                'node_id': node,
                'strategy': ag.strategy,
                'cooperation_trend': ag.strategy_trend,
                'reputation': ag.reputation,
                'betrayal_rate': ag.betrayal_rate,
            }
            for dim in OCEAN_DIMS:
                entry[dim] = ag.personality[dim]
            data.append(entry)
        return data

    def inject_defectors(self, count):
        """
        Simulate a real shock: a full psychological trauma injection.
        Shocked agents are LOCKED into defection for 30 rounds (GCN cannot
        override). Their neighbors also get a shorter lockout (cascade).
        """
        from agent import HISTORY_LEN

        cooperators = [n for n in self.agents if self.agents[n].strategy == 1]
        count = min(count, len(cooperators))
        shocked_nodes = random.sample(cooperators, count)
        neighbor_victims = set()

        for node in shocked_nodes:
            ag = self.agents[node]

            # 1. Flip strategy
            ag.strategy = 0
            self.env.update_node_state(node, 0)

            # 2. TRAUMA LOCKOUT: forced defection for 30 rounds.
            #    The GCN CANNOT override this. Simulates real radicalization
            #    where you can't just "decide" to trust again immediately.
            ag.trauma_lockout = 30

            # 3. Deep personality trauma
            ag.personality['neuroticism'] = min(0.95, ag.neuroticism + 0.35)
            ag.personality['agreeableness'] = max(0.05, ag.agreeableness - 0.35)
            ag.personality['conscientiousness'] = max(0.05, ag.conscientiousness - 0.1)

            # 4. Shift the baseline (regression anchor) toward traumatized state
            for dim in ag.personality:
                ag.baseline_personality[dim] = (
                    0.4 * ag.baseline_personality[dim] +
                    0.6 * ag.personality[dim]
                )

            # 5. Poison action history
            ag._action_history.clear()
            ag._action_history.extend([0.0] * HISTORY_LEN)

            # 6. Destroy reputation
            ag.lifetime_coops = max(0, ag.lifetime_coops - ag.lifetime_steps * 0.5)

            # 7. Shatter trust on all connected edges
            for nb in self.env.get_neighbors(node):
                if self.env.graph.has_edge(node, nb):
                    self.env.graph.edges[node, nb]['edge_trust'] = 0.0
                neighbor_victims.add(nb)

        # 8. CASCADE: neighbors suffer trauma too
        neighbor_victims -= set(shocked_nodes)
        for nb_node in neighbor_victims:
            nb_ag = self.agents[nb_node]
            nb_ag.personality['neuroticism'] = min(0.95, nb_ag.neuroticism + 0.08)
            nb_ag.personality['agreeableness'] = max(0.05, nb_ag.agreeableness - 0.05)
            # Neighbors get a short lockout too (cascade defection)
            nb_ag.trauma_lockout = max(nb_ag.trauma_lockout, 8)

    def get_random_edge_fraction(self):
        """Returns the fraction of edges that are 'random' (stranger connections)."""
        G = self.env.graph
        total = G.number_of_edges()
        if total == 0:
            return 0.0
        random_count = sum(1 for u, v in G.edges()
                          if G.edges[u, v].get('edge_type', 'local') == 'random')
        return random_count / total
