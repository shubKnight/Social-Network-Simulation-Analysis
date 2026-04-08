import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import SimulationEngine
from analytics import compute_all_metrics

st.set_page_config(layout="wide", page_title="Network Compare | Topology of Trust", page_icon="🌐")
st.title("🌐 Network Comparison")
st.markdown("Run the **same game** on four topologies. Which structures protect trust?")

TOPOLOGIES = {
    "watts_strogatz": ("🔗 Small-World", "High clustering + short paths"),
    "barabasi_albert": ("⭐ Scale-Free", "Hub-dominated (social-media-like)"),
    "erdos_renyi": ("🎲 Random", "No structure, no clusters"),
    "grid": ("📐 Grid", "Strict local connections"),
}

with st.sidebar:
    st.header("Parameters")
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    steps = st.slider("Steps", 100, 1000, 500, 50)
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)
    mutation = st.slider("Mutation (μ)", 0.0, 0.05, 0.005, 0.005)
    p_ws = st.slider("WS Randomness (p)", 0.0, 1.0, 0.0, 0.01, help="Watts-Strogatz only")
    runs = st.slider("Runs per topology", 1, 10, 3)

if st.button("🚀 Run Comparison", type="primary"):
    all_results = {}
    total = len(TOPOLOGIES) * steps * runs
    bar = st.progress(0, text="Starting...")
    done = 0
    
    for gtype, (label, _) in TOPOLOGIES.items():
        all_histories = []
        for run_i in range(runs):
            engine = SimulationEngine(n=n, k=k, p=p_ws, T=T, R=1.0, P=0.0, S=0.0,
                                       mutation_rate=mutation, init_coop_fraction=0.8,
                                       graph_type=gtype, update_rule="best_neighbor",
                                       rounds_per_update=5)
            history = []
            for s in range(steps):
                rate = engine.step()
                history.append(rate)
                done += 1
                if s % max(1, steps // 10) == 0:
                    bar.progress(min(done / total, 1.0), text=f"{label} run {run_i+1}/{runs}")
            all_histories.append(history)
        
        # Average histories across runs
        avg_history = np.mean(all_histories, axis=0).tolist()
        final_metrics = compute_all_metrics(engine.env)
        all_results[gtype] = {
            'label': label,
            'history': avg_history,
            'metrics': final_metrics,
            'final_rates': [h[-1] for h in all_histories],
        }
    
    bar.empty()
    st.session_state.compare_results = all_results
    st.rerun()

if 'compare_results' in st.session_state:
    results = st.session_state.compare_results
    
    st.subheader("📈 Cooperation Rate Over Time")
    fig = go.Figure()
    colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
    for i, (gtype, data) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            y=data['history'], mode='lines', name=data['label'],
            line=dict(color=colors[i], width=2.5),
        ))
    fig.update_layout(xaxis_title="Step", yaxis_title="Cooperation Rate",
                      yaxis=dict(range=[0, 1.05]), height=450,
                      legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Final State Comparison")
    cols = st.columns(len(results))
    for i, (gtype, data) in enumerate(results.items()):
        m = data['metrics']
        with cols[i]:
            st.markdown(f"**{data['label']}**")
            avg_final = np.mean(data['final_rates'])
            st.metric("Coop Rate", f"{avg_final:.0%}")
            st.metric("Gini", f"{m['gini_coefficient']:.3f}")
            st.metric("Clustering", f"{m['clustering_coefficient']:.3f}")
            st.metric("Avg Path", f"{m['avg_path_length']:.2f}")
            st.metric("Coop Clusters", m['num_cooperator_clusters'])
    
    with st.expander("📖 About These Topologies"):
        for _, (label, desc) in TOPOLOGIES.items():
            st.markdown(f"**{label}**: {desc}")
