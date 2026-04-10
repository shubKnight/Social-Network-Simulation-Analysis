import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import SimulationEngine
from visualization import create_network_figure

st.set_page_config(layout="wide", page_title="Resilience Lab | Topology of Trust", page_icon="💥")
st.title("💥 Resilience Lab")
st.markdown("Build trust, then shock the system. Can the network recover?")

with st.sidebar:
    st.header("Network")
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    p = st.slider("Randomness (p)", 0.0, 1.0, 0.0, 0.01)
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)
    
    st.header("Shock Parameters")
    warmup = st.slider("Warmup Steps", 50, 500, 200, 50)
    shock_size = st.slider("Defectors to Inject", 1, 50, 10, 1)
    recovery = st.slider("Recovery Steps", 50, 500, 300, 50)
    num_shocks = st.slider("Number of Shocks", 1, 5, 1)
    shock_interval = st.slider("Steps Between Shocks", 50, 200, 100, 25)

if st.button("🚀 Run Resilience Test", type="primary"):
    engine = SimulationEngine(n=n, k=k, p=p, T=T, R=1.0, P=0.0, S=0.0,
                               init_coop_fraction=0.8, temperature=0.05, temp_decay=1.0)
    
    history = []
    shock_steps = []
    total = warmup + (num_shocks * shock_interval) + recovery
    bar = st.progress(0, text="Warming up cooperative society...")
    step_count = 0
    
    for i in range(warmup):
        rate = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, warmup // 20) == 0:
            bar.progress(step_count / total, text=f"Warmup {i+1}/{warmup}")
    
    for shock_i in range(num_shocks):
        engine.inject_defectors(shock_size)
        shock_steps.append(step_count + 1)
        bar.progress(step_count / total, text=f"💥 SHOCK {shock_i+1}!")
        
        for i in range(shock_interval):
            rate = engine.step()
            step_count += 1
            history.append({'Step': step_count, 'Cooperation Rate': rate})
            if i % max(1, shock_interval // 10) == 0:
                bar.progress(step_count / total, text=f"Post-shock {shock_i+1}... {i+1}/{shock_interval}")
    
    for i in range(recovery):
        rate = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, recovery // 10) == 0:
            bar.progress(step_count / total, text=f"Recovery {i+1}/{recovery}")
    
    bar.empty()
    st.session_state.res_history = history
    st.session_state.res_shocks = shock_steps
    st.session_state.res_engine = engine
    st.session_state.res_params = {'warmup': warmup, 'shock_size': shock_size, 'p': p}
    st.rerun()

if 'res_history' in st.session_state:
    history = st.session_state.res_history
    shocks = st.session_state.res_shocks
    params = st.session_state.res_params
    
    steps_list = [h['Step'] for h in history]
    rates = [h['Cooperation Rate'] for h in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_list, y=rates, mode='lines',
                              name='Cooperation Rate', line=dict(color='#3b82f6', width=2)))
    for i, ss in enumerate(shocks):
        fig.add_vline(x=ss, line_dash="dash", line_color="red")
        fig.add_annotation(x=ss, y=1.02, text=f"💥 {i+1}", showarrow=False, font=dict(color="red"))
    fig.add_vline(x=params['warmup'], line_dash="dot", line_color="green")
    fig.update_layout(xaxis_title="Step", yaxis_title="Cooperation Rate",
                      yaxis=dict(range=[0, 1.1]), height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    pre = np.mean(rates[:params['warmup']])
    post = np.mean(rates[-50:]) if len(rates) >= 50 else np.mean(rates)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-Shock", f"{pre:.0%}")
    c2.metric("Post-Recovery", f"{post:.0%}")
    
    recov = post / pre if pre > 0 else 0
    if recov > 0.8:
        c3.metric("Recovery", "✅ Full", delta=f"{recov:.0%}")
        st.success("The network **recovered**. Trust was rebuilt.")
    elif recov > 0.3:
        c3.metric("Recovery", "⚠️ Partial", delta=f"{-(1-recov):.0%}")
        st.warning("The network **partially recovered**.")
    else:
        c3.metric("Recovery", "❌ Collapsed", delta=f"{-(1-recov):.0%}")
        st.error("Trust **collapsed permanently**.")
    
    if 'res_engine' in st.session_state:
        with st.expander("🔍 Final Network"):
            st.plotly_chart(create_network_figure(st.session_state.res_engine.env), use_container_width=True)
