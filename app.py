import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from engine import SimulationEngine
from visualization import draw_network

st.set_page_config(layout="wide", page_title="The Topology of Trust")

st.title("🔬 The Topology of Trust")
st.markdown("A multi-agent reinforcement learning simulation — *Does network structure determine whether trust survives?*")

# ── Helper: session-state backed sliders that remember values on rerun ──
def ss_slider(label, min_val, max_val, default, step, key, **kwargs):
    """A slider that persists its value in session_state across reruns."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.sidebar.slider(label, min_val, max_val, st.session_state[key], step, key=key, **kwargs)

def ss_slider_exp(label, min_val, max_val, default, step, key, **kwargs):
    """Same but for sliders inside expanders (no st.sidebar prefix)."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.slider(label, min_val, max_val, st.session_state[key], step, key=key, **kwargs)

# ── Sidebar Parameters ──
st.sidebar.header("🌐 Network Parameters")
n       = ss_slider("Number of Nodes (N)", 10, 300, 100, 10, "n_slider",
                     help="Total population.")
k       = ss_slider("Neighbors (K)", 2, 20, 6, 2, "k_slider",
                     help="Each node's initial local connections.")
p       = ss_slider("Randomness (p)", 0.0, 1.0, 0.0, 0.01, "p_slider",
                     help="0.0 = Lattice (Village) · 1.0 = Random (City)")

st.sidebar.header("🧠 Agent RL Parameters")
epsilon = ss_slider("Exploration (ε)", 0.0, 0.5, 0.02, 0.01, "eps_slider",
                     help="Chance of random action. Keep low (~0.02) for stable results.")
alpha   = ss_slider("Learning Rate (α)", 0.01, 1.0, 0.3, 0.01, "alpha_slider",
                     help="How fast agents update Q-values.")
gamma   = ss_slider("Discount Factor (γ)", 0.0, 1.0, 0.5, 0.05, "gamma_slider",
                     help="Weight of future vs immediate reward.")

st.sidebar.header("🌱 Initial Conditions")
defector_frac = ss_slider("Initial Defector %", 0.0, 0.5, 0.1, 0.05, "defector_slider",
                           help="Fraction of agents that start as defectors. Rest are cooperators.")

st.sidebar.header("🎲 Game Theory Payoffs")
with st.sidebar.expander("Payoff Matrix", expanded=True):
    st.markdown("**Constraint:** T > R > P > S")
    T = ss_slider_exp("Temptation (T)", 0.0, 3.0, 1.5, 0.1, "T_slider",
                       help="Payoff for defecting against a cooperator.")
    R = ss_slider_exp("Reward (R)", 0.0, 3.0, 1.0, 0.1, "R_slider",
                       help="Payoff for mutual cooperation.")
    P = ss_slider_exp("Punishment (P)", 0.0, 3.0, 0.1, 0.1, "P_slider",
                       help="Payoff for mutual defection.")
    S = ss_slider_exp("Sucker (S)", 0.0, 3.0, 0.0, 0.1, "S_slider",
                       help="Payoff for cooperating against a defector.")

st.sidebar.header("🔄 Imitation Dynamics")
beta = ss_slider("Imitation Strength (β)", 1.0, 30.0, 10.0, 1.0, "beta_slider",
                  help="Higher = agents are more sensitive to payoff differences when copying neighbors.")

# ── Reset ──
if st.sidebar.button("🔄 Reset Simulation", type="primary"):
    st.session_state.engine = SimulationEngine(
        n=n, k=k, p=p, alpha=alpha, gamma=gamma, epsilon=epsilon,
        T=T, R=R, P=P, S=S, imitation_strength=beta
    )
    st.session_state.history = []
    st.session_state.step_count = 0
    st.rerun()

if 'engine' not in st.session_state:
    st.session_state.engine = SimulationEngine(
        n=n, k=k, p=p, alpha=alpha, gamma=gamma, epsilon=epsilon,
        T=T, R=R, P=P, S=S, imitation_strength=beta,
        init_defector_fraction=defector_frac
    )
    st.session_state.history = []
    st.session_state.step_count = 0

engine = st.session_state.engine

# ── Layout ──
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Heatmap")
    st.markdown("🔵 **Cooperator** · 🔴 **Defector**")
    fig = draw_network(engine.env)
    st.pyplot(fig)
    plt.close(fig)
    
with col2:
    st.subheader("Controls")
    
    num_steps = st.number_input("Steps to run", min_value=1, max_value=5000, value=500)
    
    if st.button("▶️ Run Simulation", type="primary"):
        bar = st.progress(0, text="Simulating...")
        
        for i in range(num_steps):
            rate, _ = engine.step()
            st.session_state.step_count += 1
            st.session_state.history.append({
                "Step": st.session_state.step_count,
                "Cooperation Rate": rate
            })
            
            if i % max(1, num_steps // 50) == 0:
                bar.progress((i + 1) / num_steps, text=f"Step {i+1}/{num_steps}")
                
        bar.empty()
        st.rerun()
    
    st.divider()
    
    m1, m2 = st.columns(2)
    m1.metric("Step", st.session_state.step_count)
    m2.metric("Coop Rate", f"{engine.env.get_cooperation_rate():.0%}")

st.divider()

if st.session_state.history:
    st.subheader("📈 Cooperation Rate Over Time")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("Step")["Cooperation Rate"], use_container_width=True)
