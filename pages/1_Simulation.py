import streamlit as st
import numpy as np
from engine import SimulationEngine
from visualization import create_network_figure, create_cooperation_chart, create_wealth_histogram
from analytics import compute_all_metrics

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
n = ss("Nodes (N)", 10, 300, 100, 10, "n")
k = ss("Neighbors (K)", 2, 20, 6, 2, "k")
p = ss("Randomness (p)", 0.0, 1.0, 0.0, 0.01, "p")

graph_type = st.sidebar.selectbox("Graph Type", 
    ["watts_strogatz", "barabasi_albert", "erdos_renyi", "grid"],
    format_func=lambda x: {"watts_strogatz": "🔗 Small-World (Watts-Strogatz)", 
                            "barabasi_albert": "⭐ Scale-Free (Barabási-Albert)",
                            "erdos_renyi": "🎲 Random (Erdős-Rényi)",
                            "grid": "📐 Regular Grid"}[x],
    key="graph_type")

st.sidebar.header("🧠 Deep RL (PyTorch)")
learning_rate_log = ss("Learning Rate (10^x)", -5.0, -1.0, -3.0, 0.1, "lr")
learning_rate = 10 ** learning_rate_log
batch_size = ss("Batch Size", 16, 256, 64, 16, "batch_size")
gamma = ss("Discount Factor (γ)", 0.8, 0.999, 0.99, 0.001, "gamma")
temperature = ss("Initial Temperature", 0.1, 5.0, 2.0, 0.1, "temp",
                   help="High = Wide exploration (recommended: 2.0).")
temp_decay = ss("Temperature Decay", 0.9, 0.999, 0.995, 0.001, "temp_decay",
                   help="Slow decay gives the GCN time to train before committing.")
temp_warmup = ss("Warmup Steps (no decay)", 0, 200, 100, 10, "temp_warmup",
                    help="Temperature held constant for N steps so GCN fills replay buffer first.")
init_coop = ss("Initial Cooperator %", 0.0, 1.0, 0.5, 0.05, "init_coop",
                   help="50% = unbiased start.")

st.sidebar.header("🔀 Network Rewiring")
rewiring_rate = ss("Rewiring Rate", 0.0, 1.0, 0.4, 0.05, "rewiring_rate",
                   help="Fraction of exploited cooperators that cut chronic defectors & seek trustworthy replacements.")

st.sidebar.header("🎲 Payoff Matrix")
with st.sidebar.expander("T > R > P ≥ S", expanded=False):
    T = ss_exp("Temptation (T)", 0.5, 3.0, 1.1, 0.05, "T")
    R = ss_exp("Reward (R)",    0.5, 3.0, 1.0, 0.05, "R")
    P = ss_exp("Punishment (P)", 0.0, 2.0, 0.0, 0.05, "P")
    S = ss_exp("Sucker (S)",   -1.0, 1.0, -0.2, 0.05, "S")

# ── Reset ──
if st.sidebar.button("🔄 Reset Simulation", type="primary"):
    st.session_state.sim_engine = None
    st.session_state.sim_history = []
    st.session_state.sim_step = 0
    st.rerun()

def get_engine():
    if 'sim_engine' not in st.session_state or st.session_state.sim_engine is None:
        st.session_state.sim_engine = SimulationEngine(
            n=n, k=k, p=p, T=T, R=R, P=P, S=S,
            init_coop_fraction=init_coop,
            graph_type=graph_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            temperature=temperature,
            temp_decay=temp_decay,
            temp_warmup=temp_warmup,
            rewiring_rate=rewiring_rate
        )
        st.session_state.sim_history = []
        st.session_state.sim_step = 0
    return st.session_state.sim_engine

engine = get_engine()

# ── Controls ──
c1, c2, _ = st.columns([1, 1, 2])
with c1:
    num_steps = st.number_input("Steps", min_value=1, max_value=5000, value=500)
with c2:
    st.write("")
    run = st.button("▶️ Run", type="primary", use_container_width=True)

if run:
    bar = st.progress(0, text="Simulating...")
    for i in range(num_steps):
        rate = engine.step()
        st.session_state.sim_step += 1
        st.session_state.sim_history.append({
            "Step": st.session_state.sim_step,
            "Cooperation Rate": rate,
            "DQN Loss": engine.last_loss,
            "Temperature": engine.temp,
            "Rewiring Events": engine.last_rewire_count,
        })
        if i % max(1, num_steps // 50) == 0:
            bar.progress((i + 1) / num_steps, text=f"Step {i+1}/{num_steps}")
    bar.empty()
    st.rerun()

# ── Metrics ──
metrics = compute_all_metrics(engine.env)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Step", st.session_state.get('sim_step', 0))
m2.metric("Coop Rate", f"{metrics['cooperation_rate']:.0%}")
m3.metric("Gini Index", f"{metrics['gini_coefficient']:.3f}")
m4.metric("Clustering", f"{metrics['clustering_coefficient']:.3f}")
m5.metric("Avg Path", f"{metrics['avg_path_length']:.2f}")

# ── Network Graph ──
st.plotly_chart(create_network_figure(engine.env), use_container_width=True)

# ── Charts ──
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Cooperation Over Time")
    if st.session_state.get('sim_history'):
        st.plotly_chart(create_cooperation_chart(st.session_state.sim_history), use_container_width=True)
with col2:
    st.subheader("🧠 MADRL Training Stats")
    if st.session_state.get('sim_history'):
        import pandas as pd
        df = pd.DataFrame(st.session_state.sim_history)
        if "DQN Loss" in df.columns:
            st.caption("DQN Training Loss")
            st.line_chart(df.set_index("Step")["DQN Loss"])
            st.caption("Temperature (Boltzmann Exploration)")
            st.line_chart(df.set_index("Step")["Temperature"])
            st.caption("Rewiring Events per Step")
            st.line_chart(df.set_index("Step")["Rewiring Events"])
    else:
        st.info("No data yet.")
with st.expander("🔍 Cluster Analysis"):
    cc1, cc2 = st.columns(2)
    cc1.metric("Cooperator Clusters", metrics['num_cooperator_clusters'])
    cc1.metric("Largest Coop Cluster", metrics['largest_cooperator_cluster'])
    cc2.metric("Defector Clusters", metrics['num_defector_clusters'])
    cc2.metric("Largest Defect Cluster", metrics['largest_defector_cluster'])
    cc1.metric("Strategy Entropy", f"{metrics['strategy_entropy']:.3f}")
    cc2.metric("Avg Score", f"{metrics['avg_score']:.1f}")

with st.expander("🧠 Emergent Behavior Profile", expanded=True):
    profile = engine.get_behavioral_profile()
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("🤝 Chronic Cooperators", profile['chronic_cooperators'],
              help="Agents with strategy_trend > 70% (self-selected altruists)")
    e2.metric("👿 Chronic Defectors", profile['chronic_defectors'],
              help="Agents with strategy_trend < 30% (self-selected defectors)")
    e3.metric("⚖️ Swing Agents",  profile['swing_agents'],
              help="Everyone in between — opportunists, learning, transitioning")
    e4.metric("🗡️ High Betrayal", profile['high_betrayal'],
              help="Agents who cooperated often but >40% of those moves were exploited")
    st.caption(
        f"Avg Strategy Trend: {profile['avg_strategy_trend']:.2f} · "
        f"Avg Betrayal Rate: {profile['avg_betrayal_rate']:.2f} · "
        f"Avg Payoff Trend: {profile['avg_payoff_trend']:.2f}"
    )
    st.info("💡 These archetypes emerged from learning with no pre-assignment. "
            "Chronic Cooperators formed naturally in dense trust clusters; "
            "Chronic Defectors may be isolated nodes or hub exploiters.")
