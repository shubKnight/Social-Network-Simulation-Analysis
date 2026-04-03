import numpy as np
import random

class RLAgent:
    """
    A Reinforcement Learning agent that uses Q-Learning to decide whether
    to Cooperate (1) or Defect (0) in the Iterated Prisoner's Dilemma.
    
    State space: (discretized neighbor cooperation %, my last action) = 11 states
    Action space: {0: Defect, 1: Cooperate}
    """
    def __init__(self, node_id, alpha=0.3, gamma=0.5, epsilon=0.15):
        self.node_id = node_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        # State: (neighbor_coop_bucket, my_last_action)
        # neighbor_coop_bucket: 0-10 (11 bins for 0%, 10%, 20%, ..., 100%)
        # my_last_action: 0 or 1
        # Total states: 11 * 2 = 22
        self.num_states = 22
        self.num_actions = 2
        
        # Initialize Q-table with small random values to break symmetry
        self.q_table = np.random.uniform(0, 0.01, (self.num_states, self.num_actions))
        
        self.last_state = None
        self.last_action = random.choice([0, 1])
        self.cumulative_reward = 0.0

    def _get_state(self, coop_ratio):
        """Maps neighbor cooperation ratio + own last action to state index."""
        bucket = min(int(coop_ratio * 10), 10)  # 0-10
        return bucket + (self.last_action * 11)

    def choose_action(self, coop_ratio):
        """Epsilon-greedy action selection."""
        state = self._get_state(coop_ratio)
        self.last_state = state
        
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_vals = self.q_table[state]
            max_q = np.max(q_vals)
            # Break ties randomly
            best_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            action = random.choice(best_actions)
            
        self.last_action = action
        return action

    def learn(self, reward, next_coop_ratio):
        """Bellman Q-update."""
        if self.last_state is None:
            return
            
        next_state = self._get_state(next_coop_ratio)
        
        old_q = self.q_table[self.last_state, self.last_action]
        max_future_q = np.max(self.q_table[next_state])
        
        td_target = reward + self.gamma * max_future_q
        self.q_table[self.last_state, self.last_action] = old_q + self.alpha * (td_target - old_q)
        
        self.cumulative_reward += reward
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
