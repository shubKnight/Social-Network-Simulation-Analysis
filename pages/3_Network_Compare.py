import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import SimulationEngine
from analytics import compute_all_metrics

st.set_page_config(layout="wide", page_title="Network Compare | Topology of Trust", page_icon="🌐")
st.title("🌐 Network Comparison")
st.markdown("""
Run the **same Prisoner's Dilemma** on four fundamentally different network topologies 
and compare how cooperation evolves. This reveals which social structures protect trust.
""")

TOPOLOGIES = {
    "watts_strogatz": ("🔗 Small-World", "High clustering + short paths. Like a village with some long-range friendships."),
    "barabasi_albert": ("⭐ Scale-Free", "Power-law degree distribution. Like social media — few hubs, many followers."),
    "erdos_renyi": ("🎲 Random", "Uniform random connections. No structure, no clusters."),
    "grid": ("📐 Grid Lattice", "Strict local connections. Like a neighborhood where you only know your immediate neighbors."),
}

# ── Parameters ──
with st.sidebar:
    st.header("Parameters")
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    steps = st.slider("Steps", 100, 1000, 300, 50)
    T = st.slider("Temptation (T)", 1.0, 2.0, 1.4, 0.05)
    R = st.slider("Reward (R)", 0.5, 2.0, 1.0, 0.1)
    p_ws = st.slider("WS Randomness (p)", 0.0, 1.0, 0.0, 0.01, 
                      help="Only affects Watts-Strogatz topology")

if st.button("🚀 Run Comparison", type="primary"):
    all_results = {}
    total = len(TOPOLOGIES) * steps
    bar = st.progress(0, text="Starting...")
    done = 0
    
    for gtype, (label, _) in TOPOLOGIES.items():
        engine = SimulationEngine(
            n=n, k=k, p=p_ws, T=T, R=R, P=0.1, S=0.0,
            epsilon=0.02, init_defector_fraction=0.1,
            graph_type=gtype
        )
        
        history = []
        for s in range(steps):
            rate, _ = engine.step()
            history.append(rate)
            done += 1
            if s % max(1, steps // 20) == 0:
                bar.progress(done / total, text=f"Running {label}...")
        
        final_metrics = compute_all_metrics(engine.env)
        all_results[gtype] = {
            'label': label,
            'history': history,
            'metrics': final_metrics,
        }
    
    bar.empty()
    st.session_state.compare_results = all_results
    st.rerun()

# ── Display ──
if 'compare_results' in st.session_state:
    results = st.session_state.compare_results
    
    # Cooperation curves
    st.subheader("📈 Cooperation Rate Over Time")
    fig = go.Figure()
    colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
    for i, (gtype, data) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            y=data['history'],
            mode='lines',
            name=data['label'],
            line=dict(color=colors[i], width=2.5),
        ))
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.05]),
        height=450, legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    st.subheader("📊 Final State Comparison")
    cols = st.columns(len(results))
    for i, (gtype, data) in enumerate(results.items()):
        m = data['metrics']
        with cols[i]:
            st.markdown(f"**{data['label']}**")
            st.metric("Coop Rate", f"{m['cooperation_rate']:.0%}")
            st.metric("Gini", f"{m['gini_coefficient']:.3f}")
            st.metric("Clustering", f"{m['clustering_coefficient']:.3f}")
            st.metric("Avg Path", f"{m['avg_path_length']:.2f}")
            st.metric("Coop Clusters", m['num_cooperator_clusters'])
            st.metric("Largest Cluster", m['largest_cooperator_cluster'])
    
    # Topology descriptions
    with st.expander("📖 About These Topologies"):
        for gtype, (label, desc) in TOPOLOGIES.items():
            st.markdown(f"**{label}**: {desc}")
