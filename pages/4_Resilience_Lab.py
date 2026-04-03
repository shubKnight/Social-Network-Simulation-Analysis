import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import SimulationEngine
from visualization import create_network_figure

st.set_page_config(layout="wide", page_title="Resilience Lab | Topology of Trust", page_icon="💥")
st.title("💥 Resilience Lab")
st.markdown("""
**Shock Test**: Build a stable cooperative society, then inject defectors mid-simulation. 
Can the network recover? How many defectors can it absorb before trust collapses permanently?
""")

# ── Parameters ──
with st.sidebar:
    st.header("Network Setup")
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    p = st.slider("Randomness (p)", 0.0, 1.0, 0.0, 0.01)
    T = st.slider("Temptation (T)", 1.0, 2.0, 1.4, 0.05)
    
    st.header("Shock Parameters")
    warmup_steps = st.slider("Warmup Steps (build trust first)", 50, 500, 200, 50)
    shock_size = st.slider("Defectors to Inject", 1, 50, 10, 1)
    recovery_steps = st.slider("Recovery Steps (observe after shock)", 50, 500, 300, 50)
    num_shocks = st.slider("Number of Shocks", 1, 5, 1)
    shock_interval = st.slider("Steps Between Shocks", 50, 200, 100, 25)

if st.button("🚀 Run Resilience Test", type="primary"):
    engine = SimulationEngine(
        n=n, k=k, p=p, T=T, R=1.0, P=0.1, S=0.0,
        epsilon=0.02, init_defector_fraction=0.05
    )
    
    history = []
    shock_steps = []
    total_steps = warmup_steps + (num_shocks * shock_interval) + recovery_steps
    bar = st.progress(0, text="Warming up cooperative society...")
    
    step_count = 0
    
    # Phase 1: Warmup
    for i in range(warmup_steps):
        rate, _ = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, warmup_steps // 20) == 0:
            bar.progress(step_count / total_steps, text=f"Warming up... {i+1}/{warmup_steps}")
    
    # Phase 2: Shocks
    for shock_i in range(num_shocks):
        # Inject defectors
        engine.inject_defectors(shock_size)
        shock_steps.append(step_count + 1)
        bar.progress(step_count / total_steps, text=f"💥 SHOCK {shock_i+1}! Injecting {shock_size} defectors...")
        
        # Run for shock_interval steps
        for i in range(shock_interval):
            rate, _ = engine.step()
            step_count += 1
            history.append({'Step': step_count, 'Cooperation Rate': rate})
            if i % max(1, shock_interval // 10) == 0:
                bar.progress(step_count / total_steps, text=f"Post-shock {shock_i+1}... {i+1}/{shock_interval}")
    
    # Phase 3: Recovery observation
    for i in range(recovery_steps):
        rate, _ = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, recovery_steps // 10) == 0:
            bar.progress(step_count / total_steps, text=f"Observing recovery... {i+1}/{recovery_steps}")
    
    bar.empty()
    
    st.session_state.resilience_history = history
    st.session_state.resilience_shocks = shock_steps
    st.session_state.resilience_engine = engine
    st.session_state.resilience_params = {
        'warmup': warmup_steps, 'shock_size': shock_size,
        'num_shocks': num_shocks, 'p': p, 'n': n,
    }
    st.rerun()

# ── Display ──
if 'resilience_history' in st.session_state:
    history = st.session_state.resilience_history
    shocks = st.session_state.resilience_shocks
    params = st.session_state.resilience_params
    
    steps_list = [h['Step'] for h in history]
    rates = [h['Cooperation Rate'] for h in history]
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps_list, y=rates,
        mode='lines', name='Cooperation Rate',
        line=dict(color='#3b82f6', width=2),
    ))
    
    # Mark shock points
    for i, shock_step in enumerate(shocks):
        fig.add_vline(x=shock_step, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=shock_step, y=1.02, text=f"💥 Shock {i+1}",
            showarrow=False, font=dict(color="red", size=12)
        )
    
    # Mark warmup end
    fig.add_vline(x=params['warmup'], line_dash="dot", line_color="green")
    fig.add_annotation(
        x=params['warmup'], y=1.02, text="Warmup End",
        showarrow=False, font=dict(color="green", size=10)
    )
    
    fig.update_layout(
        xaxis_title="Step", yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.1]),
        height=500, title="Resilience Timeline",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis
    pre_shock = np.mean(rates[:params['warmup']])
    post_shock = np.mean(rates[-50:]) if len(rates) >= 50 else np.mean(rates)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-Shock Coop Rate", f"{pre_shock:.0%}")
    c2.metric("Post-Recovery Coop Rate", f"{post_shock:.0%}")
    
    recovery_pct = post_shock / pre_shock if pre_shock > 0 else 0
    if recovery_pct > 0.8:
        c3.metric("Recovery", "✅ Full Recovery", delta=f"{recovery_pct:.0%}")
        st.success(f"The network **recovered** from the shock! Trust was rebuilt.")
    elif recovery_pct > 0.3:
        c3.metric("Recovery", "⚠️ Partial Recovery", delta=f"{-( 1 - recovery_pct):.0%}")
        st.warning(f"The network **partially recovered**. Some trust was permanently lost.")
    else:
        c3.metric("Recovery", "❌ Collapsed", delta=f"{-(1 - recovery_pct):.0%}")
        st.error(f"The network **collapsed**. The shock created a cascade of defection that destroyed trust permanently.")
    
    # Network state
    if 'resilience_engine' in st.session_state:
        with st.expander("🔍 Final Network State"):
            st.plotly_chart(
                create_network_figure(st.session_state.resilience_engine.env, "Post-Recovery Network"),
                use_container_width=True
            )
