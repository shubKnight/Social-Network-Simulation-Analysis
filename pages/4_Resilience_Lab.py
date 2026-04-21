import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import SimulationEngine
from analytics import personality_archetype_counts, cooperation_by_personality
from visualization import (create_network_figure, _base_layout, hex_to_rgba,
                           create_personality_radar, create_personality_cooperation_bars)
from agent import OCEAN_DIMS
from theme import (apply_premium_theme, get_colors, render_mode_toggle,
                   styled_header, divider, stat_card)

st.set_page_config(layout="wide", page_title="Resilience Lab | Topology of Trust", page_icon="T")
apply_premium_theme()
render_mode_toggle()

C = get_colors()
styled_header("Resilience Lab",
              "Build trust, then shock the system with injected defectors. Can the network recover?")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p style='color:{C['accent_glow']};font-weight:600;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em'>Network</p>", unsafe_allow_html=True)
    n = st.slider("Nodes (N)", 20, 200, 100, 10)
    k = st.slider("Neighbors (K)", 2, 20, 6, 2)
    p = st.slider("Randomness (p)", 0.0, 1.0, 0.0, 0.01)
    T = st.slider("Temptation (T)", 0.5, 2.0, 1.3, 0.05)

    st.markdown(f"<p style='color:{C['defector']};font-weight:600;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;margin-top:16px'>Shock Parameters</p>", unsafe_allow_html=True)
    warmup = st.slider("Warmup Steps", 50, 500, 200, 50)
    shock_size = st.slider("Defectors to Inject", 1, 50, 10, 1)
    recovery = st.slider("Recovery Steps", 50, 500, 300, 50)
    num_shocks = st.slider("Number of Shocks", 1, 5, 1)
    shock_interval = st.slider("Steps Between Shocks", 50, 200, 100, 25)

# ── Phase Info Cards ──────────────────────────────────────────────
info_cols = st.columns(3)
info_data = [
    ("Phase 1 — Warmup", f"{warmup} steps", "Build cooperative equilibrium", C["success"]),
    ("Phase 2 — Shock", f"{num_shocks}x {shock_size} defectors", "Force strategy reversals", C["defector"]),
    ("Phase 3 — Recovery", f"{recovery} steps", "Observe self-healing capacity", C["cooperator"]),
]
for i, (title, val, desc, color) in enumerate(info_data):
    info_cols[i].markdown(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};
                border-top:3px solid {color};border-radius:10px;
                padding:18px 16px;text-align:center;box-shadow:{C['card_shadow']}">
        <div style="color:{C['text_dim']};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px">{title}</div>
        <div style="color:{C['text']};font-size:1.2rem;font-weight:700;font-family:'JetBrains Mono';margin-bottom:4px">{val}</div>
        <div style="color:{C['text_dim']};font-size:0.76rem">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

if st.button("Run Resilience Test", type="primary", use_container_width=True):
    engine = SimulationEngine(n=n, k=k, p=p, T=T, R=1.0, P=0.0, S=0.0,
                               init_coop_fraction=0.8, temperature=0.05, temp_decay=1.0)

    history = []
    shock_steps = []
    total = warmup + (num_shocks * shock_interval) + recovery
    bar = st.progress(0, text="Building cooperative society...")
    step_count = 0

    # Snapshot pre-shock personality archetypes
    pre_shock_archetypes = None

    for i in range(warmup):
        rate = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, warmup // 20) == 0:
            bar.progress(step_count / total, text=f"Warmup | step {i+1}/{warmup}")

    # Capture pre-shock state
    pre_shock_archetypes = personality_archetype_counts(engine.agents)
    pre_shock_coop_by_p = cooperation_by_personality(engine.agents)

    for shock_i in range(num_shocks):
        engine.inject_defectors(shock_size)
        shock_steps.append(step_count + 1)
        bar.progress(step_count / total, text=f"SHOCK {shock_i+1} — injecting {shock_size} defectors")

        for i in range(shock_interval):
            rate = engine.step()
            step_count += 1
            history.append({'Step': step_count, 'Cooperation Rate': rate})
            if i % max(1, shock_interval // 10) == 0:
                bar.progress(step_count / total, text=f"Post-shock {shock_i+1} | step {i+1}/{shock_interval}")

    for i in range(recovery):
        rate = engine.step()
        step_count += 1
        history.append({'Step': step_count, 'Cooperation Rate': rate})
        if i % max(1, recovery // 10) == 0:
            bar.progress(step_count / total, text=f"Recovery | step {i+1}/{recovery}")

    bar.empty()
    
    # Capture post-recovery state
    post_archetypes = personality_archetype_counts(engine.agents)
    post_coop_by_p = cooperation_by_personality(engine.agents)
    
    st.session_state.res_history = history
    st.session_state.res_shocks = shock_steps
    st.session_state.res_engine = engine
    st.session_state.res_params = {'warmup': warmup, 'shock_size': shock_size, 'p': p}
    st.session_state.res_pre_archetypes = pre_shock_archetypes
    st.session_state.res_post_archetypes = post_archetypes
    st.session_state.res_pre_coop_by_p = pre_shock_coop_by_p
    st.session_state.res_post_coop_by_p = post_coop_by_p
    st.rerun()

# ── Results ───────────────────────────────────────────────────────
if 'res_history' in st.session_state:
    history = st.session_state.res_history
    shocks = st.session_state.res_shocks
    params = st.session_state.res_params

    divider()

    steps_list = [h['Step'] for h in history]
    rates = [h['Cooperation Rate'] for h in history]

    fig = go.Figure()

    # Shade warmup zone
    fig.add_vrect(x0=0, x1=params['warmup'],
                  fillcolor=hex_to_rgba(C['success'], 0.03), line_width=0,
                  annotation_text="Warmup", annotation_position="top left",
                  annotation_font=dict(color=C["success"], size=10))

    fig.add_trace(go.Scatter(
        x=steps_list, y=rates, mode='lines', name='Cooperation Rate',
        line=dict(color=C["cooperator"], width=2.5),
        fill='tozeroy', fillcolor=hex_to_rgba(C['cooperator'], 0.06),
    ))

    for i, ss_val in enumerate(shocks):
        fig.add_vline(x=ss_val, line_dash="dash", line_color=C["defector"], line_width=2)
        fig.add_annotation(x=ss_val, y=1.05, text=f"Shock {i+1}",
                          showarrow=False, font=dict(color=C["defector"], size=10, family="Inter"))

    fig.update_layout(**_base_layout(
        xaxis_title="Step", yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.12], gridcolor=C["grid"]),
        height=480,
        title=dict(text="Resilience Timeline", font=dict(size=13, color=C["text_dim"])),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Verdict
    divider()

    pre = np.mean(rates[:params['warmup']])
    post = np.mean(rates[-50:]) if len(rates) >= 50 else np.mean(rates)
    recov = post / pre if pre > 0 else 0

    v1, v2, v3 = st.columns(3)
    v1.markdown(stat_card("Pre-Shock", f"{pre:.0%}", C["success"]), unsafe_allow_html=True)
    v2.markdown(stat_card("Post-Recovery", f"{post:.0%}", C["cooperator"]), unsafe_allow_html=True)

    if recov > 0.8:
        verdict_text, verdict_color = "Full Recovery", C["success"]
        verdict_msg = "The network recovered. Trust was rebuilt through rewiring and cooperative clustering."
    elif recov > 0.3:
        verdict_text, verdict_color = "Partial Recovery", C["warning"]
        verdict_msg = "The network partially recovered. Some trust was restored but permanent damage remains."
    else:
        verdict_text, verdict_color = "Collapsed", C["defector"]
        verdict_msg = "Trust collapsed permanently. The shock overwhelmed the network's self-healing capacity."

    v3.markdown(stat_card("Verdict", verdict_text, verdict_color), unsafe_allow_html=True)

    _v_bg = hex_to_rgba(verdict_color, 0.06)
    _v_border = hex_to_rgba(verdict_color, 0.15)
    st.markdown(f"""
    <div style="margin-top:14px;padding:14px 18px;background:{_v_bg};
                border:1px solid {_v_border};border-radius:8px;
                color:{C['text']};font-size:0.88rem;line-height:1.6">
        {verdict_msg}
    </div>
    """, unsafe_allow_html=True)

    # ── Personality Breakdown: Pre-Shock vs Post-Recovery ─────────
    if 'res_pre_archetypes' in st.session_state and 'res_post_archetypes' in st.session_state:
        divider()
        st.markdown(f"<h3 style='margin-bottom:14px;font-size:1.05rem'>Personality Response to Shock</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="padding:10px 14px;background:{hex_to_rgba(C['accent'], 0.04)};
                    border:1px solid {hex_to_rgba(C['accent'], 0.12)};border-radius:8px;
                    color:{C['text_dim']};font-size:0.8rem;line-height:1.5;margin-bottom:14px">
            Which personality archetypes survived the shock? Did community builders hold firm while
            opportunists exploited the chaos?
        </div>
        """, unsafe_allow_html=True)
        
        pre_arch = st.session_state.res_pre_archetypes
        post_arch = st.session_state.res_post_archetypes
        
        archetype_names = [
            ("Community Builders", 'community_builder', C["cooperator"]),
            ("Strategic Hubs", 'strategic_hub', C["accent_glow"]),
            ("Stoic Cooperators", 'stoic_cooperator', C["success"]),
            ("Social Butterflies", 'social_butterfly', C["warning"]),
            ("Paranoid Isolationists", 'paranoid_isolationist', C["defector"]),
            ("Opportunists", 'opportunist', C["danger"]),
        ]
        
        ac = st.columns(6)
        for i, (label, key, color) in enumerate(archetype_names):
            pre_val = pre_arch.get(key, 0)
            post_val = post_arch.get(key, 0)
            delta = post_val - pre_val
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            delta_color = C['cooperator'] if delta >= 0 and key in ('community_builder', 'strategic_hub', 'stoic_cooperator') else (
                C['defector'] if delta > 0 and key in ('paranoid_isolationist', 'opportunist') else C['text_dim']
            )
            ac[i].markdown(f"""
            <div style="background:{C['surface']};border:1px solid {C['border']};
                        border-left:3px solid {color};border-radius:10px;
                        padding:14px 12px;box-shadow:{C['card_shadow']}">
                <div style="color:{C['text_dim']};font-size:0.68rem;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px">{label}</div>
                <div style="display:flex;justify-content:space-between;align-items:baseline">
                    <span style="color:{C['text']};font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono'">{post_val}</span>
                    <span style="color:{delta_color};font-size:0.75rem;font-weight:600">{delta_str}</span>
                </div>
                <div style="color:{C['text_muted']};font-size:0.65rem;margin-top:2px">was {pre_val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Personality radar for post-recovery state
    if 'res_engine' in st.session_state:
        divider()
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Post-Recovery OCEAN Profile</p>", unsafe_allow_html=True)
            st.plotly_chart(create_personality_radar(st.session_state.res_engine.agents),
                            use_container_width=True)
        
        with r2:
            if 'res_post_coop_by_p' in st.session_state:
                st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Cooperation by Personality (Post-Recovery)</p>", unsafe_allow_html=True)
                st.plotly_chart(create_personality_cooperation_bars(st.session_state.res_post_coop_by_p),
                                use_container_width=True)
        
        with st.expander("Final Network State", expanded=False):
            st.plotly_chart(create_network_figure(st.session_state.res_engine.env,
                                                   agents=st.session_state.res_engine.agents),
                            use_container_width=True)
