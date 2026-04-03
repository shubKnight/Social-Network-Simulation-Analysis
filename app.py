import streamlit as st

st.set_page_config(layout="wide", page_title="The Topology of Trust", page_icon="🔬")

st.markdown("""
# 🔬 The Topology of Trust

### A Multi-Agent Reinforcement Learning Study on Social Networks

This project explores a fundamental question: **Does the structure of a social network 
determine whether trust can survive?**

Inspired by the *"Six Degrees of Separation"* and the *Small-World Paradox*, we simulate 
AI agents playing the Iterated Prisoner's Dilemma on different network topologies to discover 
the **mathematical tipping point** where cooperation collapses.

---

### 📖 Navigate the Experiment

| Page | What It Does |
|---|---|
| **🧪 Simulation** | Run the live simulation. Watch agents cooperate and defect in real-time on an interactive network graph. |
| **📉 Phase Transition** | Automated sweep of network randomness to find the exact tipping point where trust collapses. |
| **🌐 Network Comparison** | Compare how trust evolves on different network types: Small-World, Scale-Free, Random, and Grid. |
| **💥 Resilience Lab** | Shock a stable cooperative society by injecting defectors. Can the network recover? |

---

### 🧠 How It Works

1. **Agents** live on graph nodes. Each has a private Q-Table (RL brain).
2. **Edges** represent social connections. Agents interact with neighbors via the Prisoner's Dilemma.
3. **Learning**: Agents copy the strategy of their most successful neighbor (Spatial Imitation) 
   and occasionally explore new strategies (Q-Learning).
4. **Topology Matters**: In tight-knit communities (lattices), cooperators form "defensive clusters." 
   In random networks (like social media), shortcuts let defectors exploit distant cooperators.

> *"A few global shortcuts can destroy centuries of locally-built trust."*

---

**Built with** Python · NetworkX · NumPy · Streamlit · Plotly
""")
