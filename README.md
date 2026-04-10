# 🔬 Graph-Aware Multi-Agent Deep Reinforcement Learning (MADRL)

> *Can Neural Networks learn to trust each other based on their position in a social network?*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-green.svg)](https://networkx.org)

---

## 🧠 The Big Question

Social trust is fragile. In deep reinforcement learning, do agents learn different cooperative strategies depending on whether they live in a "Village" (high clustering, Lattice) or a "City" (Scale-Free network, influencers)?

This project is a high-fidelity **MADRL Simulation** implementing a centralized PyTorch Deep Q-Network. All 100+ agents share the same neural "brain", but are fed their local topological features (Node Degree, Clustering Coefficient) to learn context-aware strategies.

## 🏗️ Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Graph Engine** | NetworkX | Watts-Strogatz, Barabási-Albert, Erdős-Rényi, Grid |
| **Agents** | Python | Extracts topological context to feed into PyTorch |
| **Game Engine** | PyTorch | Iterated Prisoner's Dilemma with $\epsilon$-greedy exploration and Experience Replay |
| **Analytics** | NumPy + NetworkX | Gini inequality, entropy, cluster sizes |
| **Dashboard** | Streamlit + Plotly | Real-time Neural Network training visualizer and network heatmaps |

## 🔬 How It Works

Instead of hard-coded rules, the simulation uses Deep Q-Learning:
1. **Observe**: Each agent observes $S_t = [\text{last\_action}, \text{coop\_neighbor\_ratio}, \text{norm\_degree}, \text{clustering}]$
2. **Act**: The Shared PyTorch DQN predicts Q-Values. Agents Cooperate or Defect via $\epsilon$-greedy policy.
3. **Reward**: Prisoners Dilemma payoff.
4. **Learn**: Transitions $(S, A, R, S')$ are stored in a Replay Buffer. Mini-batches train the network via Mean Squared Error loss.

### Topological Awareness

Because the Neural Network receives the agent's degree and local clustering coefficient as inputs, it learns to associate its network structure with the safety of cooperation. It actively *learns* that cooperating in a highly clustered village is safe, but cooperating as a hub node is dangerous!

## 🚀 Quick Start

```bash
git clone https://github.com/shubKnight/Social-Network-Simulation-Analysis.git
cd Social-Network-Simulation-Analysis
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📖 Dashboard Features

### 🧪 1. Live Simulation
Watch the PyTorch neural network train live! Features a real-time DQN training loss curve, epsilon-decay tracking, and Plotly interactive network graph.

### 📉 2. Phase Transition Finder
Automatic network randomness sweeps (Watts-Strogatz p-value).

### 🌐 3. Network Comparison
Topology bake-off: Small-World vs Scale-Free vs Random vs Grid.

### 💥 4. Resilience Lab
Shock test: Inject 10 defectors and measure if the network recovers.
