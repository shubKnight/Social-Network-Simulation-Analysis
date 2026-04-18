# Topology of Trust
### Graph Convolutional Multi-Agent Reinforcement Learning for Cooperation Dynamics on Social Networks

---

# Problem Statement

- Social cooperation is fragile — communities, organisations, and digital platforms routinely collapse from trust to selfishness
- Classical Game Theory models this via the **Prisoner's Dilemma** but assumes agents are identical, memoryless, and on static networks
- Real social networks have **structure** (clusters, hubs), agents have **memory** (reputation), and relationships **change over time**
- **Gap**: No existing model combines graph-aware deep learning, emergent agent behaviour, and dynamic network rewiring in a single framework
- **This project bridges that gap** by building a GCN-based multi-agent RL system where agents learn cooperation strategies, build behavioural profiles from experience, and actively restructure their social connections

---

# Objectives

1. Build a **Graph Convolutional Multi-Agent RL (GC-MARL)** simulation where agents learn cooperation/defection strategies on structured social networks
2. Replace static imitation rules with a **shared GCN-DQN** that makes spatially-aware decisions using neighbourhood aggregation
3. Enable **emergent behavioural specialisation** — agents develop "personalities" (chronic cooperator, chronic defector, swing) purely from lived experience, with no pre-assignment
4. Implement **co-evolutionary dynamic rewiring** — exploited agents sever ties with defectors and seek trustworthy replacements, with the GCN seeing the restructured topology in real time
5. Study how **network topology** (Small-World, Scale-Free, Random) and **temptation level** affect the survival of cooperation

---

# Key Concepts

### Prisoner's Dilemma (PD)
Each agent plays PD with every neighbour simultaneously. Defection pays more individually, but mutual cooperation is collectively optimal.

|  | Opponent Cooperates | Opponent Defects |
|---|---|---|
| **I Cooperate** | R = 1.0, R = 1.0 | S = −0.2, T = 1.1 |
| **I Defect** | T = 1.1, S = −0.2 | P = 0.0, P = 0.0 |

*T > R > P > S must hold for a valid Prisoner's Dilemma*

### Graph Convolutional Network (GCN)
Neural network that operates directly on graph-structured data. Each layer aggregates features from neighbouring nodes:

$$H^{(l+1)} = \text{ReLU}(\hat{A} \cdot H^{(l)} \cdot W^{(l)})$$

- $\hat{A} = D^{-1}(A + I)$ — normalised adjacency with self-loops
- Layer 1: aggregates immediate neighbours → Layer 2: aggregates 2-hop neighbourhood
- Output: Q-values per node → Q(Defect) and Q(Cooperate)

### Boltzmann Exploration
Action selection via temperature-controlled softmax over Q-values:

$$P(a) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$

High τ = random exploration. Low τ = exploit learned policy. τ anneals over training.

---

# Methodology

### Phase 1 — Environment Setup
- Generate social network using NetworkX (Watts-Strogatz / Barabási-Albert / Erdős-Rényi)
- Initialise 100 agents with 50/50 cooperate/defect split
- Compute normalised adjacency matrix $\hat{A}$ for GCN input

### Phase 2 — GCN-DQN Training Loop (per step)
1. **Observe** — Build 6D state vector per agent from experience history
2. **Decide** — GCN forward pass through $\hat{A}$ → Boltzmann softmax → action
3. **Play** — Execute Prisoner's Dilemma on all edges, collect degree-weighted payoffs
4. **Learn** — Store transition in replay buffer, sample batch, update GCN via Huber Loss + Bellman target
5. **Rewire** — Exploited cooperators cut chronic defectors, seek trustworthy 2-hop replacements, recompute $\hat{A}$
6. **Anneal** — Reduce temperature (after warmup period)

### Phase 3 — Analysis
- Track cooperation rate, emergent archetypes, clustering coefficient, rewiring events per step
- Compare across topologies and temptation levels

---

# Data Flow

```
                    ┌──────────────────────────┐
                    │     Social Network       │
                    │    (NetworkX Graph)       │
                    └────────────┬─────────────┘
                                 │
                    Adjacency Matrix Â + Node Features X
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Graph Conv Layer 1     │
                    │   H = ReLU(Â · X · W₁)  │  ← aggregates 1-hop neighbours
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Graph Conv Layer 2     │
                    │   H = ReLU(Â · H · W₂)  │  ← aggregates 2-hop
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    MLP Head (32 → 2)     │
                    │   Q(Defect), Q(Cooperate)│
                    └────────────┬─────────────┘
                                 │
                    Boltzmann Softmax (τ)
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Actions Executed       │
                    │   Payoffs Collected       │
                    │   Rewards Normalised      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
          ┌─────────────────┐      ┌────────────────────┐
          │  Replay Buffer  │      │  Dynamic Rewiring  │
          │  (s, a, r, s')  │      │  Cut defector edge  │
          │  Train via       │      │  Add cooperator edge│
          │  Huber Loss      │      │  Recompute Â        │
          └─────────────────┘      └────────────────────┘
```

---

# Architecture

### Agent State Vector (6D — Emergent, Not Pre-assigned)

| # | Feature | Source | Range |
|---|---|---|---|
| 0 | `strategy` | Current action this round | {0, 1} |
| 1 | `round_payoff` | PD outcome / neighbour count | ℝ |
| 2 | `reputation` | Lifetime cooperation rate | [0, 1] |
| 3 | `strategy_trend` | Rolling mean of last 20 actions | [0, 1] |
| 4 | `payoff_trend` | Rolling normalised payoff | [−1, 1] |
| 5 | `betrayal_rate` | Fraction of coop moves exploited | [0, 1] |

### Neural Network: GraphDQN

| Layer | Type | Dims |
|---|---|---|
| Input | Node features | N × 6 |
| GC1 | Graph Convolution | 6 → 64 |
| GC2 | Graph Convolution | 64 → 64 |
| FC1 | Linear + ReLU | 64 → 32 |
| FC2 | Linear (output) | 32 → 2 |

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Learning Rate | 1e-3 | Standard for Adam |
| Batch Size | 64 | Sampled from replay buffer |
| Discount (γ) | 0.99 | Values long-run cluster survival |
| Loss | Huber (SmoothL1) | Dampens MARL non-stationarity spikes |
| Temp Warmup | 100 steps | Fill replay buffer before annealing |
| Temp Decay | 0.995 | Slow — GCN trains before committing |
| Target Sync | Every 20 steps | Stabilises Bellman targets |
| Gradient Clip | max_norm = 1.0 | Prevents exploding gradients |
| Replay Capacity | 20,000 | ~200 full-graph transitions |

---

# Co-Evolutionary Dynamic Rewiring

**Mechanism**: Exploited cooperators restructure their social ties

1. **Trigger**: Agent cooperated but was exploited by a defecting neighbour
2. **Who rewires**: Suckered agents + any cooperator adjacent to a chronic defector (betrayal_rate > 0.5)
3. **Cut**: Sever edge to the neighbour with lowest reputation
4. **Seek**: Scan 2-hop neighbourhood for best replacement (cooperating, reputation > 0.4, degree not saturated)
5. **Guard**: No cut if degree ≤ 2 (prevents isolation). No blind reconnection if no worthy candidate found
6. **Update**: Recompute $\hat{A}$ — GCN sees the new topology on the very next forward pass

**Why this matters**: Payoff is degree-weighted (`reward / n_neighbours`). Defectors who lose connections earn proportionally less per round. The network itself punishes chronic defection.

---

# Data Flow (Text Form)

1. **Graph Generation** → A social network of N nodes is generated using NetworkX. The adjacency matrix A is extracted, self-loops are added (A + I), and it is row-normalised to produce Â.

2. **State Construction** → For each agent, a 6-dimensional feature vector is assembled from its current action, round payoff, lifetime reputation, rolling strategy trend (last 20 actions), normalised payoff trend, and betrayal rate. These are stacked into a feature matrix X of shape [N × 6].

3. **GCN Forward Pass** → The matrix X is multiplied by Â in each graph convolutional layer. This performs message-passing: each node's features become a weighted aggregate of its neighbours' features. Two GCN layers mean each node "sees" up to its 2-hop neighbourhood. The result is passed through an MLP head that outputs two Q-values per node: Q(Defect) and Q(Cooperate).

4. **Action Selection** → Q-values are fed into a Boltzmann softmax function controlled by temperature τ. Early in training (high τ), actions are near-random for exploration. As τ anneals, the policy exploits learned Q-values. A warmup period holds τ constant for the first 100 steps so the replay buffer fills with diverse experience.

5. **Game Execution** → Each agent plays the Prisoner's Dilemma with every neighbour simultaneously. Payoffs are computed from the T/R/P/S matrix and normalised by the agent's degree (payoff per connection, not total), so isolated agents earn less.

6. **Experience Storage** → The transition (current_state, actions, normalised_rewards, next_state) is pushed into a replay buffer of capacity 20,000.

7. **GCN Training** → A batch of 64 transitions is sampled from the replay buffer. The Bellman target Q(s,a) = r + γ·max Q(s',a') is computed using a frozen target network. The policy network is updated via Huber Loss (SmoothL1) with gradient clipping. The target network is synced every 20 steps.

8. **Dynamic Rewiring** → Agents who were exploited (cooperated but neighbour defected) may sever the tie to their lowest-reputation defector neighbour and form a new edge to the highest-reputation cooperator within their 2-hop reach. After any rewiring, Â is recomputed so the GCN sees the new topology on its next forward pass.

9. **Loop** → Steps 2–8 repeat for the configured number of generations.

---

# Results — Current State

### What is working
- The **simulation pipeline runs end-to-end**: graph generation → GCN inference → PD payoffs → training → rewiring → adjacency recomputation functions correctly across all topology types
- The **GCN is actively learning** — training loss is non-zero, decreasing from initial values, and stabilised after switching to Huber Loss
- **Dynamic rewiring is firing** — thousands of edge rewiring events occur per run, and the clustering coefficient measurably changes (≈0.5 → ≈0.12), confirming the network is genuinely restructuring
- **Emergent behavioural differentiation occurs** — agents naturally split into chronic cooperators, chronic defectors, and swing agents purely from experience, without any pre-assigned personality types
- The **Streamlit dashboard** visualises all metrics in real time

### What is not yet producing meaningful results
- Under the standard Prisoner's Dilemma payoff matrix, **cooperation tends to decline over time** regardless of topology or hyperparameter tuning. The GCN correctly learns that defection is the individually rational strategy — which is the Nash Equilibrium of PD — but this means the system does not yet sustain high cooperation as we would hope
- The model behaves **differently under different conditions** in ways we are still investigating — for instance, Scale-Free networks sometimes sustain more cooperation than Small-World, which contradicts some existing literature, and increasing the rewiring rate does not always help
- **Temperature annealing sensitivity** remains a challenge — the GCN tends to lock into whatever strategy it has learned once temperature drops, and the exact warmup/decay schedule has outsized influence on outcomes
- Cooperation rate is **volatile but trending downward** in most runs — it peaks around 60–65% early on during exploration but settles toward 25–35% in the long run

### Honest assessment
The system successfully simulates the core dynamics and the GCN is genuinely learning from the graph structure. However, the results do not yet fully align with the sustained cooperation we hypothesised. Further research is needed into reward structures, longer training horizons, and alternative exploration strategies to produce more meaningful and interpretable outcomes.

---

# Technical Observations

| Observation | Detail |
|---|---|
| GCN Loss | Decreases and stabilises around 0.15–0.35 after Huber Loss. Previously spiked to 100+ under MSELoss. |
| Cooperation Peak | Reaches 60–65% during warmup/high-temperature phase, then declines as GCN commits to a strategy. |
| Rewiring Volume | 2,000–3,500 edge swaps over 800 steps at rate=0.4. Activity drops when cooperator candidates are exhausted. |
| Clustering Change | Starts at ~0.5 (Small-World) and drops to ~0.12 as cooperators cluster together and cut defector ties. |
| Temperature Effect | Warmup of 100 steps prevents premature lock. Without it, the GCN commits before gathering diverse experience. |
| Degree-Weighted Payoff | Creates measurable pressure on isolated defectors, but not yet strong enough to reverse cascades. |

---

# Interpretation

1. **The GCN learns the dominant strategy correctly**: The PD is designed so defection is rational. The GCN learning this is not a failure — sustaining any cooperation above 0% (the Nash Equilibrium) is itself a non-trivial result enabled by network structure and rewiring.

2. **Network structure matters, but less than expected**: Differences across topologies exist but are smaller than anticipated. This suggests payoff parameters (T, R, P, S) dominate topology effects in the current setup.

3. **Emergent specialisation validates the architecture**: Agents differentiating into behavioural clusters without pre-assignment confirms that the 6D state + GCN architecture captures meaningful agent-level variation.

4. **Rewiring reshapes the network but doesn't save cooperation on its own**: The topology visibly changes (clustering drops), but cooperators run out of candidates once defection dominates. Rewiring needs to be paired with stronger cooperative incentive structures.

5. **This is a research-in-progress result**: The framework is architecturally sound, but achieving sustained meaningful cooperation dynamics requires further work on reward signals, game formulations, and exploration schedules.

---

# Conclusion

- Built a **complete GC-MARL framework** combining Graph Convolutional Networks, Deep Q-Learning, emergent behavioural profiling, and co-evolutionary network rewiring — all running end-to-end
- The system **successfully simulates** Prisoner's Dilemma dynamics on multiple network topologies with a shared GCN performing neighbourhood-aware message passing
- **Emergent agent specialisation** works — agents develop differentiated behavioural profiles from experience without any pre-assigned types
- **Dynamic rewiring** genuinely restructures the network (measurable clustering changes) and feeds back into the GCN in real time
- However, the model **has not yet produced fully meaningful cooperation results** — the GCN correctly learns the dominant strategy (defect), and cooperation trends downward in most configurations
- **Further work** is planned: exploring alternative game formulations (Stag Hunt, Snowdrift), more nuanced reward shaping, longer training horizons, and multi-game setups to produce richer cooperative dynamics

**Key takeaway**: The framework works and the GCN learns — but making it produce sustained meaningful cooperative equilibria remains an open research challenge that we will continue investigating.

---

# Tech Stack

| Component | Technology |
|---|---|
| Neural Network | PyTorch — custom GCN layers, DQN, Huber Loss |
| Graph Engine | NetworkX — Watts-Strogatz, Barabási-Albert, Erdős-Rényi |
| Training | Experience Replay, Target Network Sync, Gradient Clipping |
| Exploration | Boltzmann Softmax + Temperature Annealing with Warmup |
| Rewiring | Co-evolutionary — reputation-based, 2-hop search, degree-weighted |
| Dashboard | Streamlit + Plotly — real-time charts and network visualisation |
| Analytics | Gini Coefficient, Shannon Entropy, Cluster Size Analysis |

---

# References

1. Kipf, T.N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
2. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533.
3. Santos, F.C. & Pacheco, J.M. (2005). *Scale-Free Networks Provide a Unifying Framework for the Emergence of Cooperation.* Physical Review Letters, 95(9).
4. Nowak, M.A. & May, R.M. (1992). *Evolutionary Games and Spatial Chaos.* Nature, 359(6398), 826–829.
5. Watts, D.J. & Strogatz, S.H. (1998). *Collective dynamics of 'small-world' networks.* Nature, 393(6684), 440–442.
6. Perolat, J. et al. (2017). *A multi-agent reinforcement learning model of common-pool resource appropriation.* NeurIPS 2017.
7. Zheng, L. et al. (2018). *MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence.* AAAI 2018.
8. Barabási, A.L. & Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science, 286(5439), 509–512.
9. Axelrod, R. (1984). *The Evolution of Cooperation.* Basic Books.
10. van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.




Layer 1 — Network Initialisation
Layer 2 — State Observation
Layer 3 — GCN Decision Making 
Layer 4 — Game Execution & Reward
Layer 5 — Learning
Layer 6 — Network Evolution


Q-Learning & The Bellman Equation

Each agent learns a function Q(s, a) that estimates the total future reward of taking action a in state s. The Bellman equation is the foundation of this learning.

The Bellman Update

$$
Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a')
$$

Symbol

Meaning

Our Value

Q(s, a)

Expected future reward for action a in state s

Output of GCN (2 values per node)

r

Immediate reward this round

Degree-weighted PD payoff, normalised to [−1, +1]

γ

Discount factor — how much the agent values the future

0.99 (values long-run survival)

maxa' Q(s', a')

Best possible future from the next state

Computed by frozen target network

How it connects to our GCN

Standard Q-learning uses a lookup table. We replace that table with a Graph Convolutional Network — the GCN takes the full graph (Â, X) and outputs Q-values for every node in one forward pass

The loss function measures how wrong the GCN's prediction was: Loss = Huber(Q_predicted − Q_target)

The GCN is trained by sampling past experiences from a replay buffer, computing the Bellman target using a frozen copy of the network, and updating weights via backpropagation

In one line: The GCN learns 'what is the long-term value of cooperating vs defecting, given my position in this specific neighbourhood?'