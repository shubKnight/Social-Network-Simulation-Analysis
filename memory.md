# Technical Evolution: The Topology of Trust

This document serves as the historical technical log and "brain memory" for the project. It tracks the structural shifts, architectural pivots, and core problems solved since inception.

---

## 1. Initial State: Pure Evolutionary Dynamics
*   **Paradigm**: Stochastic Evolutionary Game Theory (EGT).
*   **Logic**: Agents followed fixed imitation rules (Fermi distribution or "Best Neighbor").
*   **Mechanism**: If a neighbor was more successful, an agent might copy their strategy with a specific probability.
*   **Limitation**: Rigid and non-learning. The agents didn't "understand" the network; they just reacted to immediate payoffs.

## 2. Pivot I: Centralized MADRL (Multi-Agent Deep Reinforcement Learning)
*   **Problem**: Rigid rules couldn't capture complex emergent behavior or "context-aware" strategies.
*   **Solution**: Replaced stochastic rules with a **Shared PyTorch Deep Q-Network (DQN)**.
*   **Key Change**: Agents became "Neural Agents." Instead of copying, they queried a central brain to predict the best action based on features like node degree and clustering.
*   **Exploration**: Used $\epsilon$-greedy (randomly acting to gather data).

## 3. Pivot II: Graph Convolutional Networks (GC-MARL)
*   **Problem**: Standard DQNs treat agents as isolated feature vectors. They don't "see" the connections as a single physical structure.
*   **Solution**: Built a **Graph Convolutional Neural Network (GCN)** from scratch.
*   **Mathematics**: Embed the normalized Adjacency Matrix $\hat{A}$ directly into the forward pass ($H = ReLU(\hat{A} X W)$).
*   **Impact**: Trust became "spatially aware." The neural network performs message-passing, meaning it mathematically calculates aggregated neighborhood states before outputting an action.
*   **Refinement**: Replaced $\epsilon$-greedy with **Boltzmann (Softmax) Exploration**. Used "Temperature" to control exploration, preventing random actions from accidentally detonating trust clusters.

## 4. Feature Upgrade: Historical Trust & Reputation
*   **Problem**: Criticisms pointed out that real trust requires memory of neighbor behavior, not just a snapshot payoff.
*   **Solution**: Expanded the State Space to 3D: `[strategy, payoff, reputation]`.
*   **Mechanism**: A `reputation` score was added to each agent tracing their lifetime cooperation rate. The GCN aggregates these reputations, allowing agents to "infer" whether they are in a high-trust or low-trust neighborhood.

## 5. Current State: Heterogeneous Personalities & Realism Fixes (v3)
*   **The "All-Cooperation" Bug**: Previously, the simulation often locked into 100% cooperation (or defection) too early.
    *   *Cause A*: Reward Scale Bias. Cooperation in a coop-cluster earned ~6.0, while mutual defection earned 0.0. The GCN just learned "big number = good."
    *   *Cause B*: Sudden Temperature Collapse. Temperature was decaying too fast (0.95), locking the GCN into a choice before it finished training.
*   **The Realism Solution**:
    *   **Z-Score Reward Normalization**: Rewards are normalized per-step. The GCN must now learn *strategy patterns*, not just chase raw magnitudes.
    *   **Heterogeneous Personalities**: Introduced four social archetypes:
        *   🤝 **Altruist**: Gets utility bonus from neighbors cooperating; biased toward trust.
        *   😤 **Grudger**: Cooperates until betrayed, then switches to permanent defect.
        *   🦊 **Opportunist**: Pure payoff-maximizer (follows naked GCN signal).
        *   🎲 **Random**: Injects noise; occasionally ignores the GCN (15% override).
    *   **Expanded State Space (4D)**: Added `personality_embedding` to the GCN input so the neural network learns to act based on its own specific "temperament."

---

## Technical Summary of Stack
| Component | Technology |
|---|---|
| **Core** | Python, NetworkX (Graph structure) |
| **Brain** | PyTorch (Custom GCN Layers, DQN) |
| **Exploration** | Boltzmann (Softmax) with Temperature Annealing |
| **Visualization** | Streamlit (Live Dashboard), Plotly (Network rendering) |
| **Training** | Experience Replay (Memory Buffer), Periodic Target Network Sync |

---
### [Session 6] Debugging — Cooperation Collapse Fixed
- Bug: cooperation always locked at 100% or crashed to 0%
- Root causes found:
  - Grudger binary lock → 100% of grudgers permanently defected by step 30
  - Z-score normalization killed cooperation's natural payoff signal
  - State dim mismatch (engine output dim=2, GCN expected dim=3)
  - temp_decay=0.95 collapsed exploration in 60 steps before GCN trained
- Fixes:
  - Grudger → probabilistic **suspicion** float (0→1), decays each safe round
  - Soft [-1,+1] normalization instead of z-score
  - State dim unified to 4 throughout
  - temp_decay → 0.99, temp_min → 0.2, initial temp → 2.0
  - Added gradient clipping (max_norm=1.0)
- Working params: `T=1.1, S=-0.2, init_coop=0.5, temp=2.0, decay=0.99, steps=600`
- Result: 40-43% cooperation, volatile, GCN loss ~0.14-0.24 and learning

---

### [Session 7] Dynamic Network Rewiring (Co-Evolutionary)
- Added `rewiring_rate` param (default 0.3 = 30% of suckered agents attempt rewiring)
- Logic: only exploited agents rewire; cut the *specific defector* who betrayed them (lowest reputation); seek replacement in 2-hop neighbourhood (highest reputation, currently cooperating, not degree-saturated)
- If no worthy 2-hop candidate found → no rewire (no blind reconnection)
- `_recompute_A_hat()` called after each rewiring step → GCN sees new topology on next forward pass
- Edge count preserved (clean swap not add/delete)
- Verified: clustering coefficient dropped 0.532 → 0.185 over 300 steps = network genuinely reshaping
- 442 rewiring events / 300 steps with rate=0.3, T=1.1
- Dashboard: added Rewiring Rate slider + Rewiring Events per Step chart

---

### [Session 8] Removed Hardcoded Personalities — Emergent Behavior
- Stripped all personality types (Altruist/Grudger/Opportunist/Random), reward shaping, action overrides
- Each agent now builds a **behavioral profile from experience**:
  - `strategy_trend` — rolling mean of last 20 actions (chronic coop vs defect)
  - `payoff_trend` — rolling normalized payoff [-1, 1]
  - `betrayal_rate` — fraction of cooperative moves that were exploited
- State vector expanded: `[strategy, payoff, reputation, strategy_trend, payoff_trend, betrayal_rate]` (dim=6)
- GCN now learns personality differentiation purely through gradient descent
- Dashboard: replaced Personality Breakdown with **Emergent Behavior Profile** panel (Chronic Cooperators / Chronic Defectors / Swing Agents / High Betrayal counts)

---

### [Session 9] Rewiring Fixes + Payoff Mechanics
- Bugs found and fixed in rewiring:
  - `MAX_DEGREE` filter was `<` (strict) → candidates at exact max degree silently excluded → 0 rewires
  - Reputation threshold `0.55` too high for early steps → lowered to `0.4`
  - Rewiring only triggered on `was_suckered` → stopped when defectors dominated; expanded to include any cooperator adjacent to chronic defector (`betrayal_rate > 0.5`)
- Added **degree-weighted payoff**: `reward / n_neighbors` — isolated defectors earn less per round, creating real network-position pressure
- Added `temp_warmup=100`: temperature held constant for first 100 steps so GCN fills replay buffer before annealing
- Raised `gamma` default: `0.95 → 0.99` so GCN values long-run cooperation clusters
- Raised `temp_decay` default: `0.99 → 0.995` (slower annealing)
- Dashboard: added Warmup slider; updated all defaults to confirmed working params
- Current best params: `T=1.1, S=-0.2, temp=2.0, decay=0.995, warmup=100, rewiring=0.4, gamma=0.99`
- Results: 33.6% final coop, peak 62%, 3,593 rewiring events / 800 steps — realistic volatile dynamics

---

### [Session 10] Stabilizing Training Graph (Addressing Loss Spikes)
- Diagnosed massive spikes in the DQN training loss graph (reaching 100-160+).
- Cause: The non-stationary nature of MARL means target Q-values constantly shift as agent policies change. Combined with a high discount factor (`gamma=0.99`), prediction targets scale to very large numbers (~100). Standard `MSELoss` heavily penalizes and squares these large shifting prediction errors, resulting in massive outlier spikes during replay buffer batch sampling.
- Fix: Swapped `nn.MSELoss()` for `nn.SmoothL1Loss()` (Huber Loss). This treats massive prediction errors linearly rather than quadratically, clipping out the chaotic vertical spikes resulting from sudden betrayals and topology rewiring while still optimizing accurately for smaller local TD-errors.

---

### [Session 11] Trust Discount for Random Edges (Social Media Effect)
- **Insight**: "As randomness in connections increases, cooperation decreases" — mirroring social media where strangers are less trustable than physical neighbours
- **Implementation**:
  - Rewrote Watts-Strogatz generation to tag each edge as `local` (original lattice) or `random` (rewired long-range)
  - All graph types now tag edges: BA=local, ER=all random, Grid=local
  - Payoffs from `random` edges are multiplied by `STRANGER_DISCOUNT = 0.3` (strangers yield only 30% of the payoff of trusted neighbours)
  - Dynamically rewired edges are tagged `random` (new connections start as stranger ties)
  - Added `get_random_edge_fraction()` analytics method
- **Sweep results** (500 steps, T=1.1, discount=0.3):
  - p=0.0 (pure local): 37.9% coop, 19 chronic defectors
  - p=0.1: 39.7% coop, 13 chronic defectors
  - p=0.5: 29.0% coop, 57 chronic defectors ← trust collapse zone
  - p=1.0 (fully random): 32.3% coop, 37 chronic defectors
- **Finding**: The steepest cooperation drop occurs around p≈0.5 where the network loses enough local structure for trust to collapse

---

### [Session 12] Dynamic Edge Trust (Settling Trust)
- **Insight**: "Agents don't have to behave fully cooperatively with strangers *until the trust settles*." A static penalty isn't realistic — repeated interactions should build trust over time.
- **Implementation**:
  - Removed static `STRANGER_DISCOUNT`.
  - Introduced dynamic **`edge_trust`** attached to every graph edge.
  - Local edges initialise at `1.0` (fully trusted). Random edges (or dynamically rewired edges) start at `0.1` (heavy distrust).
  - During PD execution, payoffs on that specific edge are multiplied by its `edge_trust`.
  - **Trust buildup**: If both agents cooperate, `edge_trust` increases by `+0.1` (capped at 1.0).
  - **Trust shattering**: If either agent defects, `edge_trust` drops by `-0.5` (floored at 0.0).
- **Result**: Models real-world "hesitancy." A new connection yields almost no reward initially. But if agents continuously cooperate despite low payouts, the edge trust "settles" up to 1.0, restoring full payoff potential. Betrayal instantly shatters established trust.
