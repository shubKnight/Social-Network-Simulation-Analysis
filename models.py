import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class GraphConv(nn.Module):
    """
    A minimal implementation of a Graph Convolutional Layer.
    Computes: H = ReLU( A_hat * X * W )
    """
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_hat, X):
        """
        A_hat: Normalized Adjacency Matrix [Batch, N, N] or [N, N]
        X: Node features [Batch, N, in_features] or [N, in_features]
        """
        # Linear transformation
        support = torch.matmul(X, self.weight)
        # Graph convolution (message passing)
        output = torch.matmul(A_hat, support) + self.bias
        return output

class GraphDQN(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=32, action_dim=2):
        """
        Shared GCN for all agents.
        State: [strategy, last_payoff] for each node.
        Action: Q-values for [Defect (0), Cooperate (1)] for each node.
        """
        super(GraphDQN, self).__init__()
        self.gc1 = GraphConv(state_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, A_hat, X):
        # Layer 1: Learn from immediate neighbors
        x = F.relu(self.gc1(A_hat, X))
        # Layer 2: Learn from neighbors of neighbors
        x = F.relu(self.gc2(A_hat, x))
        # Final output layer per node
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """
        Since A_hat is static per environment, we only store the feature matrix X.
        state, action, reward, next_state are all arrays of shape [N, ...]
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state)
        )

    def __len__(self):
        return len(self.buffer)
