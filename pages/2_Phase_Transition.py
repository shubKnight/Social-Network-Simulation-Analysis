import streamlit as st
import numpy as np
import pandas as pd
from engine import SimulationEngine
from visualization import create_phase_transition_chart
from theme import (apply_premium_theme, get_colors, render_mode_toggle,
                   styled_header, divider, stat_card)
from visualization import hex_to_rgba

st.set_page_config(layout="wide", page_title="Phase Transition | Topology of Trust", page_icon="T")
apply_premium_theme()
render_mode_toggle()

C = get_colors()
styled_header("Phase Transition Finder",
              "Sweep network randomness (p) to locate the critical threshold where cooperation collapses")

# ── Config ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:10px;
            padding:16px 20px;margin-bottom:20px;box-shadow:{C['card_shadow']}">
    <div style="color:{C['text']};font-weight:600;font-size:0.9rem;margin-bottom:2px">Experiment Configuration</div>
    <div style="color:{C['text_dim']};font-size:0.78rem">Set parameters below then run the sweep</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
with c2:
    steps = st.slider("Steps per run", 100, 1000, 500, 50)
    runs = st.slider("Runs per p (averaging)", 1, 10, 5)
with c3:
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)
    temp = st.slider("Temperature", 0.01, 1.0, 0.05, 0.01)

granularity = st.select_slider("Sweep Granularity",
    options=["Coarse (fast)", "Medium", "Fine (slow)"], value="Medium")

GRANULARITY_MAP = {
    "Coarse (fast)": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Medium": [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0],
    "Fine (slow)": [round(i * 0.01, 2) for i in range(0, 21)] + [0.25, 0.3, 0.5, 0.75, 1.0],
}
p_values = sorted(set(GRANULARITY_MAP[granularity]))

if st.button("Run Phase Transition Sweep", type="primary", use_container_width=True):
    results = {pv: [] for pv in p_values}
    total = len(p_values) * runs
    bar = st.progress(0, text="Starting sweep...")
    done = 0

    for pv in p_values:
        for run_i in range(runs):
            engine = SimulationEngine(n=n, k=k, p=pv, T=T, R=1.0, P=0.0, S=0.0,
                                       init_coop_fraction=0.8, temperature=temp, temp_decay=1.0)
            for _ in range(steps):
                rate = engine.step()
            results[pv].append(rate)
            done += 1
            bar.progress(done / total, text=f"p = {pv:.2f} | run {run_i+1}/{runs}")

    bar.empty()
    avg_rates = [np.mean(results[pv]) for pv in p_values]
    std_rates = [np.std(results[pv]) for pv in p_values]

    st.session_state.phase_results = {
        'p_values': p_values, 'avg_rates': avg_rates, 'std_rates': std_rates,
        'params': {'n': n, 'k': k, 'T': T, 'steps': steps, 'runs': runs, 'temp': temp},
    }
    st.rerun()

# ── Results ───────────────────────────────────────────────────────
if 'phase_results' in st.session_state:
    res = st.session_state.phase_results

    divider()

    st.plotly_chart(
        create_phase_transition_chart(res['p_values'], res['avg_rates'], res['std_rates']),
        use_container_width=True
    )

    drops = [res['avg_rates'][i] - res['avg_rates'][i+1] for i in range(len(res['avg_rates'])-1)]
    if drops and max(drops) > 0.05:
        idx = np.argmax(drops)
        critical_p = (res['p_values'][idx] + res['p_values'][idx+1]) / 2

        r1, r2, r3 = st.columns(3)
        r1.markdown(stat_card("Critical Threshold", f"p ~ {critical_p:.3f}", C["defector"]), unsafe_allow_html=True)
        r2.markdown(stat_card("Max Cooperation", f"{max(res['avg_rates']):.0%}", C["cooperator"]), unsafe_allow_html=True)
        r3.markdown(stat_card("Min Cooperation", f"{min(res['avg_rates']):.0%}", C["defector"]), unsafe_allow_html=True)

        _pt_bg = hex_to_rgba(C['defector'], 0.04)
        _pt_border = hex_to_rgba(C['defector'], 0.14)
        st.markdown(f"""
        <div style="margin-top:14px;padding:12px 18px;background:{_pt_bg};
                    border:1px solid {_pt_border};border-radius:8px;
                    color:{C['text']};font-size:0.88rem">
            At <b>p ~ {critical_p:.3f}</b>, random shortcuts begin destroying cooperative clusters.
            Beyond this point, trust erodes rapidly.
        </div>
        """, unsafe_allow_html=True)

    divider()

    with st.expander("Raw Data Table"):
        df = pd.DataFrame({
            'p': res['p_values'],
            'Avg Cooperation': [f"{r:.1%}" for r in res['avg_rates']],
            'Std Dev': [f"{s:.3f}" for s in res['std_rates']],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="color:{C['text_muted']};font-size:0.75rem;text-align:center;margin-top:8px">
        N={res['params']['n']} | K={res['params']['k']} | T={res['params']['T']} |
        Temp={res['params']['temp']} | Steps={res['params']['steps']} | Runs={res['params']['runs']}
    </div>
    """, unsafe_allow_html=True)
