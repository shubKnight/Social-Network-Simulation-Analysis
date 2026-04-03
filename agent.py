import numpy as np
import random

class RLAgent:
    def __init__(self, node_id, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the Reinforcement Learning Agent.
        """
        self.node_id = node_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # Multiply epsilon by this each step
        
        # State space: discretized % of cooperating neighbors (6) x own last action (2) = 12
        self.num_states = 12 
        self.num_actions = 2 # 0: Defect, 1: Cooperate
        
        # Initialize Q-table with slightly optimistic values to encourage exploration initially
        # rather than zeros.
        self.q_table = np.ones((self.num_states, self.num_actions)) * 1.5
        
        self.last_state = None
        self.last_action = random.choice([0, 1])

    def _discretize_state(self, coop_percentage):
        """Converts a continuous percentage and previous action into a discrete state index."""
        if coop_percentage == 0:
            env_state = 0
        elif coop_percentage <= 0.2:
            env_state = 1
        elif coop_percentage <= 0.4:
            env_state = 2
        elif coop_percentage <= 0.6:
            env_state = 3
        elif coop_percentage <= 0.8:
            env_state = 4
        else:
            env_state = 5
            
        # State incorporates 'what I did last time' + 'what neighbors did'
        return env_state + (self.last_action * 6)

    def choose_action(self, coop_percentage):
        """Chooses an action using epsilon-greedy policy based on current state."""
        state = self._discretize_state(coop_percentage)
        self.last_state = state
        
        # Exploration
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        # Exploitation
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            action = random.choice(np.where(q_values == max_q)[0])
            
        self.last_action = action
        return action

    def learn(self, next_coop_percentage, reward):
        """Updates the Q-table using the Bellman equation."""
        if self.last_state is None or self.last_action is None:
            return 
            
        next_state = self._discretize_state(next_coop_percentage)
        
        current_q = self.q_table[self.last_state, self.last_action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[self.last_state, self.last_action] = new_q
        
        # Decay epsilon so they eventually settle down instead of exploring forever
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_table(self):
        return self.q_table
