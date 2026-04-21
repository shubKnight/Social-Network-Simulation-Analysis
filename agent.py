"""
Neural Agent — OCEAN Multi-Attribute Personality Model

Each agent has a 5-dimensional continuous personality vector based on the
Big Five (OCEAN) model from psychology:

  Openness          — exploration breadth (scales per-agent Boltzmann temperature)
  Conscientiousness — long-term strategic discipline (scales effective γ / Q-sharpness)
  Extraversion      — social reach (controls rewiring aggression and max degree)
  Agreeableness     — forgiveness / trust (gates rewiring, speeds trust rebuild)
  Neuroticism       — emotional reactivity (amplifies recent payoff swings)

These 5 intrinsic traits combine with 6 emergent behavioral features
(strategy, payoff, reputation, strategy_trend, payoff_trend, betrayal_rate)
to form an 11-dimensional state vector fed into the shared GCN.

Personality is assigned at birth (normal distribution centered at 0.5) and
remains fixed — the GCN learns differentiated strategies purely through
gradient descent over the personality landscape.
"""

import numpy as np
from collections import deque

HISTORY_LEN = 20   # rolling window length

# ── Personality dimension names (canonical order) ─────────────────
OCEAN_DIMS = ['openness', 'agreeableness', 'conscientiousness',
              'extraversion', 'neuroticism']


class NeuralAgent:
    def __init__(self, node_id, strategy=1, max_payoff=6.0, personality=None):
        """
        Args:
            node_id      : Node ID in the network graph.
            strategy     : 1 = Cooperate, 0 = Defect (initial).
            max_payoff   : Rough max possible payoff per round (used for normalization).
            personality  : Optional dict of OCEAN values. If None, randomly generated.
        """
        self.node_id = node_id
        self.strategy = strategy
        self.max_payoff = max(max_payoff, 1.0)

        # ── OCEAN Personality Vector ──────────────────────────────
        # Each dimension ∈ [0, 1], drawn from N(0.5, 0.2) clipped.
        # Bell-curve distribution: most agents are moderate, few are extreme.
        if personality is not None:
            self.personality = personality
        else:
            self.personality = {
                dim: float(np.clip(np.random.normal(0.5, 0.2), 0.0, 1.0))
                for dim in OCEAN_DIMS
            }

        # Round-level accounting
        self.round_payoff      = 0.0
        self.cumulative_payoff = 0.0

        # Rolling behavioral history (emergent — not pre-assigned)
        self._action_history = deque([strategy] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._payoff_history = deque([0.0]    * HISTORY_LEN, maxlen=HISTORY_LEN)

        # Betrayal tracking (needed for strategy_trend & rewiring)
        self._coop_moves   = 1.0   # how many times this agent cooperated
        self._betrayed     = 0.0   # how many of those were exploited

        # Reputation (lifetime cooperation rate)
        self.lifetime_steps = 1.0
        self.lifetime_coops = 1.0 if strategy == 1 else 0.0

        # Suckered flag (set each round, cleared on reset)
        self.was_suckered = False

    # ------------------------------------------------------------------
    # Personality accessors
    # ------------------------------------------------------------------

    @property
    def personality_vec(self):
        """Returns personality as a numpy array in canonical OCEAN order."""
        return np.array([self.personality[d] for d in OCEAN_DIMS], dtype=np.float32)

    @property
    def openness(self):
        return self.personality['openness']

    @property
    def agreeableness(self):
        return self.personality['agreeableness']

    @property
    def conscientiousness(self):
        return self.personality['conscientiousness']

    @property
    def extraversion(self):
        return self.personality['extraversion']

    @property
    def neuroticism(self):
        return self.personality['neuroticism']

    # ------------------------------------------------------------------
    # Personality-derived behavioral modifiers
    # ------------------------------------------------------------------

    def effective_temperature(self, base_temp):
        """
        Per-agent exploration temperature.
        High-openness agents explore more (higher temp multiplier).
        Low-openness agents exploit learned strategies faster.
        Range: base_temp * [0.5 .. 1.5]
        """
        return base_temp * (0.5 + self.openness)

    @property
    def forgiveness_threshold(self):
        """
        How tolerant this agent is of betrayal before rewiring.
        High-agreeableness → higher threshold (forgives more, rewires less).
        Low-agreeableness → lower threshold (vengeful, rewires aggressively).
        Range: [0.15 .. 0.65]
        """
        return 0.15 + 0.5 * self.agreeableness

    @property
    def q_sharpness(self):
        """
        Multiplier for Q-values before softmax during action selection.
        High-conscientiousness → sharper decisions (more exploitation of policy).
        Low-conscientiousness → more impulsive (flatter Q-value differences).
        Range: [0.8 .. 1.2]
        """
        return 0.8 + 0.4 * self.conscientiousness

    @property
    def cooperation_propensity(self):
        """
        Direct additive bias on Q-values: positive = bias toward cooperation,
        negative = bias toward defection.

        This is the KEY mechanism that makes personality matter. The shared GCN
        produces nearly identical Q-values for all agents. This bias shifts
        the decision boundary *before* softmax so that personality directly
        determines whether an agent leans cooperative or defensive.

        Components:
          Agreeableness (+)  : Agreeable agents intrinsically prefer cooperation.
          Conscientiousness (+) : Disciplined agents value long-term mutual benefit.
          Neuroticism (-)    : Neurotic agents fear betrayal → lean defect.
          Openness (+/-)     : Open agents are willing to take cooperative risks.

        Range: roughly [-1.5 .. +1.5] for extreme personalities.
        Typical (0.5 on all): 0.0 (no bias — pure GCN policy).
        """
        bias = (
            (self.agreeableness - 0.5) * 1.5 +       # strongest driver
            (self.conscientiousness - 0.5) * 0.8 +    # values long-term gains
            -(self.neuroticism - 0.5) * 1.2 +         # fear of betrayal
            (self.openness - 0.5) * 0.3               # risk tolerance
        )
        return float(bias)

    @property
    def max_preferred_degree(self):
        """
        Maximum number of connections this agent is comfortable with.
        High-extraversion → hub-seeking (more connections).
        Low-extraversion → prefers tight, small clusters.
        Range: [4 .. 16]
        """
        return int(4 + 12 * self.extraversion)

    @property
    def rewire_probability(self):
        """
        Base probability of attempting a rewire when eligible.
        Extraversion increases willingness, agreeableness decreases it.
        Range: [0.05 .. 0.85]
        """
        # High extraversion → wants to rewire (seek connections)
        # High agreeableness → forgives instead of rewiring
        raw = 0.3 + 0.4 * self.extraversion - 0.25 * self.agreeableness
        return float(np.clip(raw, 0.05, 0.85))

    # ------------------------------------------------------------------
    # Personality distance (for homophily-based rewiring)
    # ------------------------------------------------------------------

    def personality_distance(self, other):
        """
        Normalized Euclidean distance between this agent's personality and
        another's. Returns a float in [0, 1] where 0 = identical, 1 = maximally
        different (opposite corners of the 5D unit hypercube).

        Max possible Euclidean distance in 5D unit cube = sqrt(5) ≈ 2.236
        """
        diff = self.personality_vec - other.personality_vec
        return float(np.linalg.norm(diff) / np.sqrt(len(OCEAN_DIMS)))

    def personality_similarity(self, other):
        """1 - distance. Higher = more similar. Range [0, 1]."""
        return 1.0 - self.personality_distance(other)

    # ------------------------------------------------------------------
    # Emergent behavioral profile (the "personality" the GCN learns)
    # ------------------------------------------------------------------

    @property
    def strategy_trend(self):
        """Rolling cooperation rate over last HISTORY_LEN steps (0→1)."""
        return sum(self._action_history) / HISTORY_LEN

    @property
    def payoff_trend(self):
        """Rolling mean payoff, soft-normalized to [-1, 1]."""
        raw = sum(self._payoff_history) / HISTORY_LEN
        return max(-1.0, min(1.0, raw / self.max_payoff))

    @property
    def weighted_payoff_trend(self):
        """
        Neuroticism-weighted payoff trend.
        High-neuroticism agents' trend is disproportionately influenced by
        the most recent rounds (exponential recency weighting).
        """
        if not self._payoff_history:
            return 0.0
        # Decay factor: high neuroticism → heavier recency bias
        alpha = 0.5 + 0.4 * self.neuroticism   # [0.5 .. 0.9]
        weights = np.array([alpha ** (HISTORY_LEN - 1 - i) for i in range(len(self._payoff_history))])
        weights /= weights.sum()
        raw = np.dot(weights, list(self._payoff_history))
        return float(max(-1.0, min(1.0, raw / self.max_payoff)))

    @property
    def betrayal_rate(self):
        """Fraction of cooperative moves that were exploited (0→1)."""
        if self._coop_moves < 1.0:
            return 0.0
        return min(1.0, self._betrayed / self._coop_moves)

    @property
    def reputation(self):
        """Lifetime cooperation rate (0→1)."""
        return self.lifetime_coops / self.lifetime_steps

    # ------------------------------------------------------------------
    # Per-round lifecycle
    # ------------------------------------------------------------------

    def reset_round_payoff(self):
        self.round_payoff = 0.0
        self.was_suckered = False

    def add_payoff(self, payoff, opponent_action=None):
        """Record payoff from one interaction this round."""
        self.round_payoff      += payoff
        self.cumulative_payoff += payoff

        # Track betrayal: we cooperated but opponent defected
        if self.strategy == 1 and opponent_action == 0:
            self.was_suckered = True

    def update_history(self, action):
        """Call once per round after action is finalized."""
        self._action_history.append(float(action))
        self._payoff_history.append(self.round_payoff)

        # Reputation update
        self.lifetime_steps += 1.0
        if action == 1:
            self.lifetime_coops += 1.0

        # Betrayal stats update
        if action == 1:
            self._coop_moves += 1.0
            if self.was_suckered:
                self._betrayed += 1.0
