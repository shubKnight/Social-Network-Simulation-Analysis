"""
Neural Agent — Emergent Behavioral Profile

Agents have NO pre-assigned personality. Instead, each agent maintains a rolling
behavioral history computed from their own lived experience. This profile is fed
into the shared GCN as part of the state vector, allowing the network to learn
differentiated strategies per agent type — purely through gradient descent.

Emergent behavioral features (replace personality_embed):
  strategy_trend   — rolling mean of last 20 actions (0=chronic defector, 1=chronic cooperator)
  payoff_trend     — rolling mean of recent payoffs, normalized by max possible
  betrayal_rate    — fraction of cooperative moves that were exploited by neighbors
"""

from collections import deque

HISTORY_LEN = 20   # rolling window length


class NeuralAgent:
    def __init__(self, node_id, strategy=1, max_payoff=6.0):
        """
        Args:
            node_id    : Node ID in the network graph.
            strategy   : 1 = Cooperate, 0 = Defect (initial).
            max_payoff : Rough max possible payoff per round (used for normalization).
        """
        self.node_id = node_id
        self.strategy = strategy
        self.max_payoff = max(max_payoff, 1.0)

        # Round-level accounting
        self.round_payoff    = 0.0
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
