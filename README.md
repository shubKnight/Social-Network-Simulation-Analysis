# 🔬 The Topology of Trust

> *Does the structure of a social network determine whether cooperation survives or collapses?*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-green.svg)](https://networkx.org)

---

## 🧠 The Big Question

Trust thrives in tight-knit communities but collapses when global "shortcuts" (like social media) are introduced. This project uses **Pure Evolutionary Game Theory** on networks to find the exact **mathematical tipping point** where cooperation collapses.

## 🏗️ Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Graph Engine** | NetworkX | Watts-Strogatz, Barabási-Albert, Erdős-Rényi, Grid networks |
| **Agents** | Python | Minimal evolutionary agents with strategy + payoff |
| **Game Engine** | NumPy | IPD with Nowak & May spatial imitation + Fermi dynamics |
| **Analytics** | NumPy + NetworkX | Gini coefficient, entropy, cluster analysis |
| **Dashboard** | Streamlit + Plotly | 4-page interactive research dashboard |

## 🔬 How It Works

Each agent has a fixed strategy (Cooperate or Defect). Every round:
1. **Play**: All agents play the Prisoner's Dilemma with their neighbors
2. **Imitate**: Each agent copies the strategy of their most successful neighbor (Nowak & May 1992)
3. **Mutate**: Small chance of random strategy flip (prevents absorbing states)

**No reinforcement learning, no Q-tables** — pure evolutionary dynamics.

### Why Topology Matters

In a **ring lattice** (Village): cooperators form clusters where interior cooperators earn 6×R = 6.0,
outperforming boundary defectors who earn ~4.2. The cluster expands.

In a **random network** (City): no clusters form, defectors exploit shortcuts to reach fresh cooperators.

## 🚀 Quick Start

```bash
git clone https://github.com/shubKnight/Social-Network-Simulation-Analysis.git
cd Social-Network-Simulation-Analysis
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📖 Dashboard Pages

### 🧪 1. Live Simulation
Interactive Plotly network graph, Gini inequality, clustering metrics, wealth distribution.

### 📉 2. Phase Transition Finder
Automated p-sweep producing the classic S-curve with critical threshold detection.

### 🌐 3. Network Comparison
Side-by-side IPD on Small-World, Scale-Free, Random, and Grid topologies.

### 💥 4. Resilience Lab
Shock test: build trust → inject defectors → measure recovery.

## 🔬 Key Findings

| Network | Cooperation | Why |
|---|---|---|
| **Small-World (p=0)** | ~83% | Tight clusters protect cooperators |
| **Grid Lattice** | ~65% | Local structure sustained cooperation |
| **Scale-Free** | ~0% | Hub nodes amplify defection cascades |
| **Random** | ~35% | No stable clusters; bimodal outcomes |

## 📁 Project Structure

```
TopologyOfTrust/
├── app.py                     # Landing page
├── pages/
│   ├── 1_Simulation.py        # Live simulation
│   ├── 2_Phase_Transition.py  # p-sweep
│   ├── 3_Network_Compare.py   # Topology comparison
│   └── 4_Resilience_Lab.py    # Shock testing
├── engine.py                  # Evolutionary game engine
├── agent.py                   # Minimal evolutionary agent
├── environment.py             # Multi-topology graph engine
├── analytics.py               # Gini, entropy, cluster analysis
├── visualization.py           # Plotly visualizations
└── requirements.txt
```

## 📚 References

- Nowak, M.A. & May, R.M. (1992). *Evolutionary games and spatial chaos.* Nature.
- Santos, F.C. & Pacheco, J.M. (2005). *Scale-free networks and cooperation.* PRL.
- Watts, D.J. & Strogatz, S.H. (1998). *Collective dynamics of 'small-world' networks.* Nature.
- Szabó, G. & Fáth, G. (2007). *Evolutionary games on graphs.* Physics Reports.
