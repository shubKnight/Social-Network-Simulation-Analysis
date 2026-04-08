import streamlit as st
import numpy as np
from engine import SimulationEngine
from visualization import create_phase_transition_chart

st.set_page_config(layout="wide", page_title="Phase Transition | Topology of Trust", page_icon="📉")
st.title("📉 Phase Transition Finder")
st.markdown("""
Sweep network randomness (p) to find the **critical threshold** where cooperation collapses.
""")

c1, c2, c3 = st.columns(3)
with c1:
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
with c2:
    steps = st.slider("Steps per run", 100, 1000, 500, 50)
    runs = st.slider("Runs per p (for averaging)", 1, 10, 5)
with c3:
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)
    mutation = st.slider("Mutation Rate (μ)", 0.0, 0.05, 0.005, 0.005)

granularity = st.select_slider("Sweep Granularity",
    options=["Coarse (fast)", "Medium", "Fine (slow)"], value="Medium")

GRANULARITY_MAP = {
    "Coarse (fast)": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Medium": [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Fine (slow)": [round(i * 0.01, 2) for i in range(0, 21)] + [0.25, 0.3, 0.5, 0.75, 1.0],
}
p_values = sorted(set(GRANULARITY_MAP[granularity]))

if st.button("🚀 Run Phase Transition Sweep", type="primary"):
    results = {pv: [] for pv in p_values}
    total = len(p_values) * runs
    bar = st.progress(0, text="Starting...")
    done = 0
    
    for pv in p_values:
        for run_i in range(runs):
            engine = SimulationEngine(n=n, k=k, p=pv, T=T, R=1.0, P=0.0, S=0.0,
                                       mutation_rate=mutation, init_coop_fraction=0.8,
                                       update_rule="best_neighbor", rounds_per_update=5)
            for _ in range(steps):
                rate = engine.step()
            results[pv].append(rate)
            done += 1
            bar.progress(done / total, text=f"p={pv:.2f}, run {run_i+1}/{runs}")
    
    bar.empty()
    avg_rates = [np.mean(results[pv]) for pv in p_values]
    std_rates = [np.std(results[pv]) for pv in p_values]
    
    st.session_state.phase_results = {
        'p_values': p_values, 'avg_rates': avg_rates, 'std_rates': std_rates,
        'params': {'n': n, 'k': k, 'T': T, 'steps': steps, 'runs': runs, 'mutation': mutation},
    }
    st.rerun()

if 'phase_results' in st.session_state:
    res = st.session_state.phase_results
    
    st.plotly_chart(
        create_phase_transition_chart(res['p_values'], res['avg_rates'], res['std_rates']),
        use_container_width=True
    )
    
    drops = [res['avg_rates'][i] - res['avg_rates'][i+1] for i in range(len(res['avg_rates'])-1)]
    if drops and max(drops) > 0.05:
        idx = np.argmax(drops)
        critical_p = (res['p_values'][idx] + res['p_values'][idx+1]) / 2
        st.success(f"🎯 **Critical Threshold: p ≈ {critical_p:.3f}**")
        st.markdown("At this point, random shortcuts begin destroying cooperative clusters.")
    
    with st.expander("📊 Raw Data"):
        import pandas as pd
        df = pd.DataFrame({
            'p': res['p_values'],
            'Avg Coop': [f"{r:.1%}" for r in res['avg_rates']],
            'Std': [f"{s:.3f}" for s in res['std_rates']],
        })
        st.dataframe(df, use_container_width=True)
    
    st.caption(f"N={res['params']['n']}, K={res['params']['k']}, T={res['params']['T']}, "
               f"μ={res['params']['mutation']}, Steps={res['params']['steps']}, Runs={res['params']['runs']}")
