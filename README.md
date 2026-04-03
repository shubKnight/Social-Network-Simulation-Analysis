# 🔬 The Topology of Trust

> *A Multi-Agent Reinforcement Learning simulation exploring how social network structure determines whether cooperation survives or collapses.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-green.svg)](https://networkx.org)

---

## 🧠 The Big Question

> *In a hyper-connected world, does increased connectivity make society less cooperative?*

This project tests the **Small-World Paradox**: trust thrives in tight-knit communities but collapses when global "shortcuts" (like social media connections) are introduced. We find the exact **mathematical tipping point** where cooperation suddenly dies.

## 🏗️ Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Graph Engine** | NetworkX | Generates Watts-Strogatz, Barabási-Albert, Erdős-Rényi, and Grid networks |
| **Agent Brains** | NumPy (Q-Learning) | Each agent has an independent Q-table for decision-making |
| **Game Engine** | Custom Python | Iterated Prisoner's Dilemma with Spatial Imitation dynamics |
| **Analytics** | NumPy + NetworkX | Gini coefficient, clustering, entropy, cluster analysis |
| **Dashboard** | Streamlit + Plotly | 4-page interactive research dashboard |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/shubKnight/Social-Network-Simulation-Analysis.git
cd Social-Network-Simulation-Analysis

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

## 📖 Dashboard Pages

### 🧪 1. Live Simulation
Run the simulation in real-time with an interactive network graph. Watch agents cooperate (blue) and defect (red). Track Gini inequality, clustering coefficient, and cooperator cluster sizes.

### 📉 2. Phase Transition Finder
Automated sweep of network randomness (p) to find the **critical threshold** where cooperation collapses. Produces the classic S-curve used in statistical physics.

### 🌐 3. Network Comparison
Side-by-side comparison of 4 network topologies:
- **Small-World** (Watts-Strogatz): High clustering + short paths
- **Scale-Free** (Barabási-Albert): Power-law degree distribution (social media-like)
- **Random** (Erdős-Rényi): No structure
- **Grid Lattice**: Strict local connections

### 💥 4. Resilience Lab
"Shock test" — build a stable cooperative society, then inject defectors. Measure whether the network recovers, partially heals, or permanently collapses.

## 🔬 Key Findings

- **Critical Threshold**: Cooperation collapses at approximately **p ≈ 0.05** — just 5% network randomness destroys trust.
- **Cluster Defense**: In lattice networks (p=0), cooperators form defensive clusters that out-earn boundary defectors.
- **Social Media Effect**: Scale-free networks (Barabási-Albert) are particularly vulnerable because hub nodes amplify defection cascades.

## 🎓 Theoretical Background

This project combines three fields:

1. **Graph Theory** — Watts-Strogatz small-world model, clustering coefficient, path length
2. **Game Theory** — Iterated Prisoner's Dilemma, Nash equilibrium, spatial games
3. **Reinforcement Learning** — Q-Learning, epsilon-greedy exploration, Bellman equation

The strategy update mechanism follows **Nowak & May (1992)**: agents copy the strategy of their most successful neighbor, with Q-Learning providing intelligent exploration.

## 📁 Project Structure

```
TopologyOfTrust/
├── app.py                  # Main entry point (landing page)
├── pages/
│   ├── 1_Simulation.py     # Live interactive simulation
│   ├── 2_Phase_Transition.py  # Automated p-sweep
│   ├── 3_Network_Compare.py   # Topology comparison
│   └── 4_Resilience_Lab.py    # Shock testing
├── engine.py               # Simulation engine (IPD + Spatial Imitation)
├── agent.py                # Q-Learning RL agent
├── environment.py          # Network graph (multi-topology support)
├── analytics.py            # Gini, entropy, cluster analysis
├── visualization.py        # Plotly interactive visualizations
├── requirements.txt
└── README.md
```

## 📚 References

- Watts, D.J. & Strogatz, S.H. (1998). *Collective dynamics of 'small-world' networks.* Nature.
- Nowak, M.A. & May, R.M. (1992). *Evolutionary games and spatial chaos.* Nature.
- Santos, F.C. & Pacheco, J.M. (2005). *Scale-free networks provide a unifying framework for the emergence of cooperation.* PRL.
- Szabó, G. & Fáth, G. (2007). *Evolutionary games on graphs.* Physics Reports.

---

*Built with Python, NetworkX, Streamlit, and Plotly.*
