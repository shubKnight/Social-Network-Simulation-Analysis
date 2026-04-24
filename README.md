# Topology of Trust

**Personality-Driven Multi-Agent Reinforcement Learning on Evolving Social Networks**

A simulation framework that models the emergence of cooperation and defection in social networks using Graph Convolutional Networks (GCNs), the OCEAN personality model from psychology, and co-evolutionary network dynamics.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-green.svg)](https://networkx.org)

---

## Overview

Social trust is a function of both individual psychology and network structure. This project investigates how cooperative and defective behaviours emerge, cluster, and evolve when autonomous agents are embedded in realistic social topologies.

Each agent in the simulation is governed by:
- A **shared Graph Convolutional Network** that performs 3-hop neighbourhood aggregation, allowing agents to perceive cooperation patterns across their extended social graph.
- An **intrinsic OCEAN personality profile** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) drawn from a Dirichlet-constrained budget, creating natural cognitive tradeoffs.
- A **dynamic personality drift** mechanism where traits evolve based on social experiences — betrayal, cooperation, and novelty — with regression toward a stable baseline.

Agents play an iterated Prisoner's Dilemma, rewire their connections based on personality homophily, and learn strategies through Deep Q-Learning. The result is a self-organising system where personality-driven echo chambers, trust clusters, and defection cascades emerge without any hardcoded rules.

---

## Architecture

| Component | Technology | Role |
|---|---|---|
| Graph Engine | NetworkX | Watts-Strogatz, Barabasi-Albert, Erdos-Renyi, and Grid topologies |
| Neural Agents | Python | OCEAN personality model, behavioural tracking, dynamic drift |
| GCN-DQN | PyTorch | 3-layer Graph Convolutional Network with MLP head for Q-value estimation |
| Game Engine | Custom | Iterated Prisoner's Dilemma with dynamic edge trust and degree-weighted payoffs |
| Analytics | NumPy, NetworkX | Gini coefficient, strategy entropy, personality assortativity, archetype classification |
| Dashboard | Streamlit, Plotly | Real-time simulation control, network visualisation, and training analytics |

---

## How It Works

### 1. Observation

Each agent constructs an 11-dimensional state vector combining behavioural features (strategy, payoff, reputation, strategy trend, weighted payoff trend, betrayal rate) with its intrinsic OCEAN personality traits. The GCN aggregates these features across the local neighbourhood via message-passing on the normalised adjacency matrix.

### 2. Decision

The shared GCN-DQN outputs Q-values for each agent. Actions are sampled via Boltzmann (softmax) exploration, where the temperature is modulated per-agent by Openness — more open agents explore more broadly. A cooperation propensity bias derived from personality traits shifts the Q-values before sampling.

### 3. Interaction

Agents play the Prisoner's Dilemma against all neighbours. Payoffs are scaled by a dynamic `edge_trust` value on each connection. Mutual cooperation builds trust incrementally; defection shatters it. Payoffs are normalised by degree to prevent hub nodes from accumulating disproportionate rewards.

### 4. Adaptation

After each round:
- Personality traits drift based on round events (e.g., being suckered raises neuroticism, mutual cooperation raises agreeableness).
- Traits regress 15% toward a birth baseline to prevent population-wide psychological collapse.
- Exploited agents may rewire connections, seeking personality-similar neighbours (homophily-driven clustering).
- Transitions are stored in a replay buffer and used to train the GCN via Smooth L1 (Huber) loss.

---

## Key Mechanisms

### OCEAN Personality Model

Each trait directly modulates agent behaviour at the algorithmic level:

| Trait | Mechanism |
|---|---|
| Openness | Scales per-agent Boltzmann temperature (exploration breadth) |
| Conscientiousness | Scales effective discount factor (long-term vs. short-term discipline) |
| Extraversion | Controls maximum degree and rewiring aggression |
| Agreeableness | Gates rewiring threshold and speeds trust recovery |
| Neuroticism | Amplifies payoff trend reactivity (emotional volatility) |

Traits are allocated via a Dirichlet-constrained budget (sum = 2.5), forcing agents to make cognitive tradeoffs — high openness must be "paid for" by lower conscientiousness, and so on.

### Dynamic Edge Trust

Edges carry a continuous `edge_trust` value between 0 and 1. Local (lattice) edges initialise at 1.0; random (rewired) edges start at 0.1. Payoffs on each edge are multiplied by its trust value. Mutual cooperation builds trust by +0.1 per round; any defection drops it by -0.5. This models the real-world asymmetry where trust is slow to build and fast to break.

### Trauma Lockout (Resilience Testing)

The Resilience Lab can inject defectors as a controlled shock. Shocked agents undergo:
1. Forced defection for 30 rounds (GCN policy override)
2. Deep personality trauma (neuroticism spike, agreeableness collapse)
3. Baseline anchor shift (60% toward traumatised state)
4. Action history poisoning and trust destruction on all edges
5. Cascade: neighbours receive an 8-round lockout and minor personality trauma

This produces realistic, visible shock-and-recovery dynamics rather than instant reversion.

---

## Dashboard

The application provides four interactive pages:

### 1. Live Simulation
Real-time control of a running simulation with configurable network parameters, payoff matrix, and RL hyperparameters. Visualisations include the interactive network graph (colourable by strategy or any OCEAN dimension), cooperation rate timeline, DQN training loss, temperature decay, rewiring activity, personality radar charts, and archetype classification.

### 2. Phase Transition Finder
Automated sweep across network randomness (Watts-Strogatz rewiring probability). Identifies the critical threshold where cooperation collapses as the network transitions from local lattice to random graph.

### 3. Network Comparison
Side-by-side topology comparison across Small-World, Scale-Free, Random, and Grid networks under identical parameters. Reveals how network structure alone determines the viability of cooperation.

### 4. Resilience Lab
Controlled shock-and-recovery experiments. Builds a cooperative equilibrium during warmup, then injects defectors with configurable shock size and frequency. Dual-axis timeline chart shows both cooperation rate and raw cooperator count to make shock impact clearly visible.

---

## Project Structure

```
TopologyOfTrust/
  About.py              # Streamlit landing page
  agent.py              # NeuralAgent class, OCEAN model, personality drift
  analytics.py          # Metrics: Gini, entropy, assortativity, archetypes
  engine.py             # SimulationEngine: GCN training, PD payoffs, rewiring
  environment.py        # SocialNetwork: graph generation, edge/node management
  models.py             # GraphConv, GraphDQN, ReplayBuffer (PyTorch)
  visualization.py      # Plotly chart generators (network, radar, bars, etc.)
  theme.py              # Light/dark theme system for Streamlit
  memory.md             # Technical evolution log
  pages/
    1_Simulation.py     # Live simulation dashboard
    2_Phase_Transition.py
    3_Network_Compare.py
    4_Resilience_Lab.py
```

---

## Quick Start

```bash
git clone <repository-url>
cd TopologyOfTrust
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run About.py
```

---

## Requirements

- Python 3.9+
- PyTorch
- Streamlit
- NetworkX
- Plotly
- NumPy

See `requirements.txt` for exact versions.

---

## References

- Watts, D.J. and Strogatz, S.H. (1998). Collective dynamics of small-world networks. *Nature*, 393(6684), pp.440-442.
- Barabasi, A.L. and Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), pp.509-512.
- Costa, P.T. and McCrae, R.R. (1992). *Revised NEO Personality Inventory (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual*. Psychological Assessment Resources.
- Kipf, T.N. and Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.
- Nowak, M.A. and May, R.M. (1992). Evolutionary games and spatial chaos. *Nature*, 359(6398), pp.826-829.
