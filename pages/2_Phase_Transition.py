import streamlit as st
import numpy as np
from engine import SimulationEngine
from visualization import create_phase_transition_chart

st.set_page_config(layout="wide", page_title="Phase Transition | Topology of Trust", page_icon="📉")
st.title("📉 Phase Transition Finder")
st.markdown("""
Find the **critical threshold** of network randomness where cooperation collapses.
This sweep runs the simulation at different values of *p* and plots the resulting cooperation rate.
""")

# ── Parameters ──
c1, c2, c3 = st.columns(3)
with c1:
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
with c2:
    steps = st.slider("Steps per run", 100, 1000, 300, 50)
    runs = st.slider("Runs per p (for averaging)", 1, 5, 3)
with c3:
    T = st.slider("Temptation (T)", 1.0, 2.0, 1.4, 0.05)
    R = st.slider("Reward (R)", 0.5, 2.0, 1.0, 0.1)
    P = st.slider("Punishment (P)", 0.0, 1.0, 0.1, 0.05)

granularity = st.select_slider(
    "Sweep Granularity",
    options=["Coarse (fast)", "Medium", "Fine (slow)"],
    value="Medium"
)

GRANULARITY_MAP = {
    "Coarse (fast)": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Medium": [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Fine (slow)": [round(i * 0.01, 2) for i in range(0, 21)] + [0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
}

p_values = sorted(set(GRANULARITY_MAP[granularity]))

if st.button("🚀 Run Phase Transition Sweep", type="primary"):
    results = {p_val: [] for p_val in p_values}
    total_work = len(p_values) * runs
    bar = st.progress(0, text="Starting sweep...")
    done = 0
    
    for p_val in p_values:
        for run_i in range(runs):
            engine = SimulationEngine(
                n=n, k=k, p=p_val, T=T, R=R, P=P, S=0.0,
                epsilon=0.02, init_defector_fraction=0.1
            )
            for _ in range(steps):
                rate, _ = engine.step()
            results[p_val].append(rate)
            done += 1
            bar.progress(done / total_work, text=f"p={p_val:.2f}, run {run_i+1}/{runs}")
    
    bar.empty()
    
    avg_rates = [np.mean(results[p_val]) for p_val in p_values]
    std_rates = [np.std(results[p_val]) for p_val in p_values]
    
    st.session_state.phase_results = {
        'p_values': p_values,
        'avg_rates': avg_rates,
        'std_rates': std_rates,
        'params': {'n': n, 'k': k, 'T': T, 'R': R, 'P': P, 'steps': steps, 'runs': runs},
    }
    st.rerun()

# ── Display Results ──
if 'phase_results' in st.session_state:
    res = st.session_state.phase_results
    
    st.plotly_chart(
        create_phase_transition_chart(res['p_values'], res['avg_rates'], res['std_rates']),
        use_container_width=True
    )
    
    # Find critical threshold
    drops = [res['avg_rates'][i] - res['avg_rates'][i+1] for i in range(len(res['avg_rates'])-1)]
    if drops:
        max_drop_idx = np.argmax(drops)
        critical_p = (res['p_values'][max_drop_idx] + res['p_values'][max_drop_idx+1]) / 2
        
        st.success(f"🎯 **Critical Threshold: p ≈ {critical_p:.3f}**")
        st.markdown(f"""
        At this point, the network has just enough random shortcuts to destroy the 
        cooperative clusters that protect trust. Below this threshold, cooperators 
        thrive in tight-knit communities. Above it, defectors exploit the shortcuts 
        to invade and collapse the entire society.
        """)
    
    # Data table
    with st.expander("📊 Raw Data"):
        import pandas as pd
        df = pd.DataFrame({
            'Randomness (p)': res['p_values'],
            'Avg Cooperation': [f"{r:.1%}" for r in res['avg_rates']],
            'Std Dev': [f"{s:.3f}" for s in res['std_rates']],
        })
        st.dataframe(df, use_container_width=True)
    
    st.caption(f"Parameters: N={res['params']['n']}, K={res['params']['k']}, "
               f"T={res['params']['T']}, R={res['params']['R']}, P={res['params']['P']}, "
               f"Steps={res['params']['steps']}, Runs={res['params']['runs']}")
