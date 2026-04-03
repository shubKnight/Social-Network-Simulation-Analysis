import streamlit as st
import numpy as np
from engine import SimulationEngine
from visualization import create_network_figure, create_cooperation_chart, create_wealth_histogram
from analytics import compute_all_metrics, gini_coefficient

st.set_page_config(layout="wide", page_title="Simulation | Topology of Trust", page_icon="🧪")
st.title("🧪 Live Simulation")

# ── Session-state backed sliders ──
def ss(label, mn, mx, default, step, key, **kw):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.sidebar.slider(label, mn, mx, st.session_state[key], step, key=key, **kw)

def ss_exp(label, mn, mx, default, step, key, **kw):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.slider(label, mn, mx, st.session_state[key], step, key=key, **kw)

# ── Sidebar ──
st.sidebar.header("🌐 Network")
n       = ss("Nodes (N)", 10, 300, 100, 10, "n")
k       = ss("Neighbors (K)", 2, 20, 6, 2, "k")
p       = ss("Randomness (p)", 0.0, 1.0, 0.0, 0.01, "p")

graph_type = st.sidebar.selectbox("Graph Type", 
    options=["watts_strogatz", "barabasi_albert", "erdos_renyi", "grid"],
    format_func=lambda x: {"watts_strogatz": "🔗 Small-World (Watts-Strogatz)", 
                            "barabasi_albert": "⭐ Scale-Free (Barabási-Albert)",
                            "erdos_renyi": "🎲 Random (Erdős-Rényi)",
                            "grid": "📐 Regular Grid"}[x],
    key="graph_type")

st.sidebar.header("🧠 RL Parameters")
epsilon = ss("Exploration (ε)", 0.0, 0.3, 0.02, 0.01, "eps")
alpha   = ss("Learning Rate (α)", 0.01, 1.0, 0.3, 0.01, "alpha")
gamma   = ss("Discount (γ)", 0.0, 1.0, 0.5, 0.05, "gamma")

st.sidebar.header("🌱 Initial State")
defector_frac = ss("Defector %", 0.0, 0.5, 0.1, 0.05, "def_frac")

st.sidebar.header("🎲 Payoff Matrix")
with st.sidebar.expander("T > R > P > S", expanded=False):
    T = ss_exp("Temptation (T)", 0.0, 3.0, 1.4, 0.1, "T")
    R = ss_exp("Reward (R)", 0.0, 3.0, 1.0, 0.1, "R")
    P = ss_exp("Punishment (P)", 0.0, 3.0, 0.1, 0.1, "P")
    S = ss_exp("Sucker (S)", 0.0, 3.0, 0.0, 0.1, "S")

# ── Reset ──
if st.sidebar.button("🔄 Reset Simulation", type="primary"):
    st.session_state.sim_engine = None
    st.session_state.sim_history = []
    st.session_state.sim_step = 0
    st.rerun()

def get_engine():
    if 'sim_engine' not in st.session_state or st.session_state.sim_engine is None:
        st.session_state.sim_engine = SimulationEngine(
            n=n, k=k, p=p, alpha=alpha, gamma=gamma, epsilon=epsilon,
            T=T, R=R, P=P, S=S, init_defector_fraction=defector_frac,
            graph_type=graph_type
        )
        st.session_state.sim_history = []
        st.session_state.sim_step = 0
    return st.session_state.sim_engine

engine = get_engine()

# ── Controls ──
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    num_steps = st.number_input("Steps", min_value=1, max_value=5000, value=500)
with c2:
    st.write("")  # spacing
    run = st.button("▶️ Run", type="primary", use_container_width=True)

if run:
    bar = st.progress(0, text="Simulating...")
    for i in range(num_steps):
        rate, _ = engine.step()
        st.session_state.sim_step += 1
        st.session_state.sim_history.append({
            "Step": st.session_state.sim_step,
            "Cooperation Rate": rate,
        })
        if i % max(1, num_steps // 50) == 0:
            bar.progress((i + 1) / num_steps, text=f"Step {i+1}/{num_steps}")
    bar.empty()
    st.rerun()

# ── Metrics Row ──
metrics = compute_all_metrics(engine.env)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Step", st.session_state.get('sim_step', 0))
m2.metric("Coop Rate", f"{metrics['cooperation_rate']:.0%}")
m3.metric("Gini Index", f"{metrics['gini_coefficient']:.3f}")
m4.metric("Clustering", f"{metrics['clustering_coefficient']:.3f}")
m5.metric("Avg Path", f"{metrics['avg_path_length']:.2f}")

# ── Network Graph ──
st.plotly_chart(
    create_network_figure(engine.env, title="Agent Network"),
    use_container_width=True
)

# ── Charts ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Cooperation Over Time")
    if st.session_state.get('sim_history'):
        st.plotly_chart(
            create_cooperation_chart(st.session_state.sim_history),
            use_container_width=True
        )
    else:
        st.info("Click **Run** to start the simulation.")

with col2:
    st.subheader("💰 Wealth Distribution")
    scores = engine.env.get_scores()
    if any(s > 0 for s in scores):
        st.plotly_chart(create_wealth_histogram(scores), use_container_width=True)
    else:
        st.info("No data yet.")

# ── Cluster Info ──
with st.expander("🔍 Cluster Analysis"):
    cc1, cc2 = st.columns(2)
    cc1.metric("Cooperator Clusters", metrics['num_cooperator_clusters'])
    cc1.metric("Largest Coop Cluster", metrics['largest_cooperator_cluster'])
    cc2.metric("Defector Clusters", metrics['num_defector_clusters'])
    cc2.metric("Largest Defect Cluster", metrics['largest_defector_cluster'])
    cc1.metric("Strategy Entropy", f"{metrics['strategy_entropy']:.3f}")
    cc2.metric("Avg Score", f"{metrics['avg_score']:.1f}")
