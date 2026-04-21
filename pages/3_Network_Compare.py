import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import SimulationEngine
from analytics import compute_all_metrics, personality_assortativity
from visualization import _base_layout, hex_to_rgba
from agent import OCEAN_DIMS
from theme import (apply_premium_theme, get_colors, get_chart_colors,
                   render_mode_toggle, styled_header, divider, stat_card)

st.set_page_config(layout="wide", page_title="Network Compare | Topology of Trust", page_icon="T")
apply_premium_theme()
render_mode_toggle()

C = get_colors()
CC = get_chart_colors()
styled_header("Network Comparison",
              "Run the same game on four topologies — which structures protect trust?")

TOPOLOGIES = {
    "watts_strogatz": ("Small-World", "High clustering + short paths", CC[0]),
    "barabasi_albert": ("Scale-Free", "Hub-dominated topology", CC[1]),
    "erdos_renyi": ("Random", "No structure, no clusters", CC[2]),
    "grid": ("Grid Lattice", "Strict local connections", CC[3]),
}

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='color:{C['accent_glow']};font-weight:600;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em'>Parameters</p>", unsafe_allow_html=True)
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    steps = st.slider("Steps", 100, 1000, 500, 50)
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)
    temp = st.slider("Temperature", 0.01, 1.0, 0.05, 0.01)
    p_ws = st.slider("WS Randomness (p)", 0.0, 1.0, 0.0, 0.01, help="Watts-Strogatz only")
    runs = st.slider("Runs per topology", 1, 10, 3)

# ── Topology cards ────────────────────────────────────────────────
cols = st.columns(4)
for i, (gtype, (label, desc, color)) in enumerate(TOPOLOGIES.items()):
    with cols[i]:
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};
                    border-top:3px solid {color};border-radius:10px;
                    padding:16px 14px;text-align:center;min-height:100px;
                    box-shadow:{C['card_shadow']}">
            <div style="color:{C['text']};font-weight:700;font-size:0.9rem;margin-bottom:4px">{label}</div>
            <div style="color:{C['text_dim']};font-size:0.76rem;line-height:1.4">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

if st.button("Run Comparison", type="primary", use_container_width=True):
    all_results = {}
    total = len(TOPOLOGIES) * steps * runs
    bar = st.progress(0, text="Starting...")
    done = 0

    for gtype, (label, _, color) in TOPOLOGIES.items():
        all_histories = []
        for run_i in range(runs):
            engine = SimulationEngine(n=n, k=k, p=p_ws, T=T, R=1.0, P=0.0, S=0.0,
                                       init_coop_fraction=0.8, graph_type=gtype,
                                       temperature=temp, temp_decay=1.0)
            history = []
            for s in range(steps):
                rate = engine.step()
                history.append(rate)
                done += 1
                if s % max(1, steps // 10) == 0:
                    bar.progress(min(done / total, 1.0), text=f"{label} | run {run_i+1}/{runs}")
            all_histories.append(history)

        avg_history = np.mean(all_histories, axis=0).tolist()
        final_metrics = compute_all_metrics(engine.env, agents=engine.agents)
        
        # Get personality assortativity for final state
        p_assort = personality_assortativity(engine.env.graph, engine.agents)
        avg_assort = np.mean(list(p_assort.values()))
        
        all_results[gtype] = {
            'label': label, 'color': color,
            'history': avg_history,
            'metrics': final_metrics,
            'final_rates': [h[-1] for h in all_histories],
            'personality_assortativity': p_assort,
            'avg_assortativity': avg_assort,
        }

    bar.empty()
    st.session_state.compare_results = all_results
    st.rerun()

# ── Results ───────────────────────────────────────────────────────
if 'compare_results' in st.session_state:
    results = st.session_state.compare_results

    divider()

    fig = go.Figure()
    for gtype, data in results.items():
        fig.add_trace(go.Scatter(
            y=data['history'], mode='lines', name=data['label'],
            line=dict(color=data['color'], width=2.5),
        ))
    fig.update_layout(**_base_layout(
        xaxis_title="Step", yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.05], gridcolor=C["grid"]),
        height=460, legend=dict(x=0.01, y=0.99),
        title=dict(text="Cooperation Rate Over Time", font=dict(size=13, color=C["text_dim"])),
    ))
    st.plotly_chart(fig, use_container_width=True)

    divider()

    st.markdown(f"<h3 style='margin-bottom:14px;font-size:1.05rem'>Final State Comparison</h3>", unsafe_allow_html=True)
    cols = st.columns(len(results))
    for i, (gtype, data) in enumerate(results.items()):
        m = data['metrics']
        avg_final = np.mean(data['final_rates'])
        avg_assort = data.get('avg_assortativity', 0)
        assort_color = C['cooperator'] if avg_assort > 0 else C['defector']
        with cols[i]:
            st.markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-top:3px solid {data['color']};border-radius:10px;
                        padding:18px 14px;text-align:center;box-shadow:{C['card_shadow']}">
                <div style="color:{C['text']};font-weight:700;font-size:0.9rem;margin-bottom:10px">{data['label']}</div>
                <div style="color:{data['color']};font-size:1.8rem;font-weight:800;font-family:'JetBrains Mono'">{avg_final:.0%}</div>
                <div style="color:{C['text_dim']};font-size:0.7rem;margin-bottom:10px">Final Cooperation</div>
                <div style="color:{C['text_dim']};font-size:0.76rem;line-height:1.8">
                    Gini: <b style="color:{C['text']}">{m['gini_coefficient']:.3f}</b><br>
                    Clustering: <b style="color:{C['text']}">{m['clustering_coefficient']:.3f}</b><br>
                    Avg Path: <b style="color:{C['text']}">{m['avg_path_length']:.2f}</b><br>
                    Coop Clusters: <b style="color:{C['text']}">{m['num_cooperator_clusters']}</b><br>
                    <span style="border-top:1px solid {C['border']};display:inline-block;padding-top:6px;margin-top:4px">
                    Personality Echo: <b style="color:{assort_color}">{avg_assort:+.3f}</b>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
