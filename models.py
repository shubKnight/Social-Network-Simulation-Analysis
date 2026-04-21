import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.
    Computes: H = ReLU( A_hat @ X @ W + b )
    
    The key insight: by multiplying X by A_hat first, each node's 
    features become a weighted average of its neighbors' features.
    This is how the GCN "sees" the local topology.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_hat, X):
        """
        A_hat : [N, N] or [B, N, N]  — normalized adjacency matrix
        X     : [N, F] or [B, N, F]  — node feature matrix
        """
        support = torch.matmul(X, self.weight)         # [..., N, out]
        return torch.matmul(A_hat, support) + self.bias


class GraphDQN(nn.Module):
    """
    Shared GCN-DQN for all agents.

    State per node (dim=11):
        Behavioral (emergent from experience):
          [strategy, last_payoff, reputation,
           strategy_trend, payoff_trend, betrayal_rate]

        Personality (intrinsic OCEAN traits):
          [openness, agreeableness, conscientiousness,
           extraversion, neuroticism]

    The GCN aggregates neighbor features through message-passing,
    then an MLP head estimates Q-values: [Q(Defect), Q(Cooperate)].

    Architecture: 3 GCN layers (3-hop receptive field) + 2-layer MLP head.
    The 3rd GCN layer extends the receptive field so agents can detect
    personality clusters and cooperation patterns further out in the graph.
    """
    def __init__(self, state_dim=11, hidden_dim=128, action_dim=2):
        super().__init__()
        # 3 GCN layers: 1-hop → 2-hop → 3-hop neighborhood aggregation
        self.gc1 = GraphConv(state_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3 = GraphConv(hidden_dim, hidden_dim)

        # Dropout for regularization (MARL can overfit to replay buffer patterns)
        self.dropout = nn.Dropout(0.1)

        # MLP head for Q-value estimation
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, A_hat, X):
        # Layer 1: aggregate immediate neighbours
        x = F.relu(self.gc1(A_hat, X))
        # Layer 2: aggregate 2-hop neighbourhood
        x = F.relu(self.gc2(A_hat, x))
        # Layer 3: aggregate 3-hop neighbourhood (captures personality clusters)
        x = F.relu(self.gc3(A_hat, x))
        x = self.dropout(x)
        # MLP head
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """
        state / next_state : np.ndarray [N, state_dim]
        action             : np.ndarray [N]  int64
        reward             : np.ndarray [N]  float32
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
        )

    def __len__(self):
        return len(self.buffer)
