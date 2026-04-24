import streamlit as st
import numpy as np
import pandas as pd
from engine import SimulationEngine
from visualization import (create_network_figure, create_cooperation_chart,
                           create_training_chart, create_personality_radar,
                           create_personality_cooperation_bars,
                           create_personality_distribution,
                           create_assortativity_chart,
                           hex_to_rgba, OCEAN_LABELS)
from analytics import compute_all_metrics
from agent import OCEAN_DIMS
from theme import (apply_premium_theme, get_colors, render_mode_toggle,
                   styled_header, divider, stat_card, section_label)

st.set_page_config(layout="wide", page_title="Simulation | Topology of Trust", page_icon="T")
apply_premium_theme()

C = get_colors()
styled_header("Live Simulation", "Real-time multi-agent cooperation dynamics on evolving networks")

# ── Session-state backed sliders ──────────────────────────────────
def ss(label, mn, mx, default, step, key, **kw):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.sidebar.slider(label, mn, mx, st.session_state[key], step, key=key, **kw)

def ss_exp(label, mn, mx, default, step, key, **kw):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.slider(label, mn, mx, st.session_state[key], step, key=key, **kw)

# ── Sidebar ───────────────────────────────────────────────────────
render_mode_toggle()

section_label("Network Topology")
n = ss("Nodes (N)", 10, 300, 100, 10, "n")
k = ss("Neighbors (K)", 2, 20, 6, 2, "k")
p = ss("Randomness (p)", 0.0, 1.0, 0.0, 0.01, "p",
       help="Controls edge randomness. High values mimic social media. Press Reset to apply.")
graph_type = st.sidebar.selectbox("Graph Type",
    ["watts_strogatz", "barabasi_albert", "erdos_renyi", "grid"],
    format_func=lambda x: {"watts_strogatz": "Small-World (WS)",
                            "barabasi_albert": "Scale-Free (BA)",
                            "erdos_renyi": "Random (ER)",
                            "grid": "Grid Lattice"}[x],
    key="graph_type")

section_label("Deep RL Parameters")
learning_rate_log = ss("Learning Rate (10^x)", -5.0, -1.0, -3.0, 0.1, "lr")
learning_rate = 10 ** learning_rate_log
batch_size = ss("Batch Size", 16, 256, 64, 16, "batch_size")
gamma = ss("Discount Factor", 0.8, 0.999, 0.99, 0.001, "gamma")
temperature = ss("Initial Temperature", 0.1, 5.0, 2.0, 0.1, "temp",
                   help="Exploration breadth. Scaled per-agent by Openness.")
temp_decay = ss("Temperature Decay", 0.9, 0.999, 0.995, 0.001, "temp_decay")
temp_warmup = ss("Warmup Steps", 0, 200, 100, 10, "temp_warmup")
init_coop = ss("Initial Cooperator %", 0.0, 1.0, 0.5, 0.05, "init_coop")

section_label("Network Rewiring")
rewiring_rate = ss("Rewiring Rate", 0.0, 1.0, 0.4, 0.05, "rewiring_rate",
                   help="Base rate — modulated per-agent by Extraversion & Agreeableness.")

section_label("Payoff Matrix")
with st.sidebar.expander("T > R > P >= S", expanded=False):
    T = ss_exp("Temptation (T)", 0.5, 3.0, 1.5, 0.05, "T")
    R = ss_exp("Reward (R)",    0.5, 3.0, 1.0, 0.05, "R")
    P = ss_exp("Punishment (P)", 0.0, 2.0, 0.1, 0.05, "P")
    S = ss_exp("Sucker (S)",   -1.0, 1.0, -0.3, 0.05, "S")

section_label("Visualization")
color_by = st.sidebar.selectbox(
    "Color Graph By",
    ["strategy"] + list(OCEAN_DIMS),
    format_func=lambda x: "Strategy (Cooperate/Defect)" if x == "strategy" else OCEAN_LABELS.get(x, x),
    key="color_by",
    help="Color network nodes by strategy or any OCEAN personality dimension.",
)

# ── State Tracking ────────────────────────────────────────────────
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

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

# ── Helper ────────────────────────────────────────────────────────
def _record_step(rate):
    st.session_state.sim_step += 1
    st.session_state.sim_history.append({
        "Step": st.session_state.sim_step,
        "Cooperation Rate": rate,
        "DQN Loss": engine.last_loss,
        "Temperature": engine.temp,
        "Rewiring Events": engine.last_rewire_count,
        "Stranger %": engine.get_random_edge_fraction(),
    })

# ── Control Bar ───────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background:{C['surface']};
    border:1px solid {C['border']};
    border-radius:10px;
    padding:14px 18px 6px;
    margin-bottom:16px;
    box-shadow:{C['card_shadow']};
">
    <div style="color:{C['text']};font-weight:600;font-size:0.9rem;margin-bottom:8px">Controls</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1, 1, 1])

if c1.button("Run Live", type="primary", use_container_width=True):
    st.session_state.is_playing = True
    st.rerun()

if c2.button("Pause", use_container_width=True):
    st.session_state.is_playing = False
    st.rerun()

if c3.button("+10 Steps", use_container_width=True):
    st.session_state.is_playing = False
    for _ in range(10):
        _record_step(engine.step())
    st.rerun()

if c4.button("+50 Steps", use_container_width=True):
    st.session_state.is_playing = False
    for _ in range(50):
        _record_step(engine.step())
    st.rerun()

if c5.button("Reset", use_container_width=True):
    st.session_state.sim_engine = None
    st.session_state.sim_history = []
    st.session_state.sim_step = 0
    st.session_state.is_playing = False
    st.rerun()

# Fast forward & speed settings
f1, f2, f3 = st.columns([1, 1.5, 3])
with f1:
    fast_steps = st.number_input("Steps", min_value=1, max_value=5000, value=500,
                                  label_visibility="collapsed")
with f2:
    live_speed = st.selectbox("Live Speed", ["Smooth (1 step/frame)", "Balanced (5 steps/frame)", "Fast (20 steps/frame)"], 
                              index=1, label_visibility="collapsed")
    SPEED_MAP = {"Smooth (1 step/frame)": 1, "Balanced (5 steps/frame)": 5, "Fast (20 steps/frame)": 20}
    st.session_state.live_batch_size = SPEED_MAP[live_speed]

with f3:
    if st.button("Fast Forward", use_container_width=True):
        st.session_state.is_playing = False
        bar = st.progress(0, text="Running...")
        for i in range(fast_steps):
            _record_step(engine.step())
            if i % max(1, fast_steps // 50) == 0:
                bar.progress((i + 1) / fast_steps, text=f"Step {i+1}/{fast_steps}")
        bar.empty()
        st.rerun()

if st.session_state.is_playing:
    _ind_bg = hex_to_rgba(C['accent'], 0.05)
    _ind_border = hex_to_rgba(C['accent'], 0.2)
    st.markdown(f"""
    <div style="
        background:{_ind_bg};
        border:1px solid {_ind_border};
        border-radius:8px;
        padding:8px 14px;
        text-align:center;
        color:{C['accent_glow']};
        font-weight:600;
        font-size:0.85rem;
    ">Simulation running live — press Pause to stop</div>
    """, unsafe_allow_html=True)

# ── Metrics Dashboard ─────────────────────────────────────────────
divider()

metrics = compute_all_metrics(engine.env, agents=engine.agents)
step_n = st.session_state.get('sim_step', 0)
stranger_frac = engine.get_random_edge_fraction()

m_cols = st.columns(6)
metric_data = [
    ("Step", step_n, C["text"]),
    ("Cooperation", f"{metrics['cooperation_rate']:.0%}", C["cooperator"]),
    ("Gini Index", f"{metrics['gini_coefficient']:.3f}", C["warning"]),
    ("Clustering", f"{metrics['clustering_coefficient']:.3f}", C["accent_glow"]),
    ("Avg Path", f"{metrics['avg_path_length']:.2f}", C["success"]),
    ("Stranger Edges", f"{stranger_frac:.1%}", C["defector"]),
]
for i, (label, value, color) in enumerate(metric_data):
    m_cols[i].markdown(stat_card(label, value, color), unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────
divider()

# View toggle
if 'show_all' not in st.session_state:
    st.session_state.show_all = False

toggle_col1, toggle_col2 = st.columns([4, 1])
with toggle_col2:
    if st.session_state.show_all:
        if st.button("Tabbed View", use_container_width=True, key="_view_toggle"):
            st.session_state.show_all = False
            st.rerun()
    else:
        if st.button("Show All", use_container_width=True, key="_view_toggle"):
            st.session_state.show_all = True
            st.rerun()

with toggle_col1:
    mode_label = "Full Dashboard" if st.session_state.show_all else "Tabbed"
    st.markdown(f"""
    <div style="color:{C['text_dim']};font-size:0.78rem;padding:8px 0;font-weight:500">
        View: <span style="color:{C['accent_glow']};font-weight:600">{mode_label}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: render sections ───────────────────────────────────────

def _render_network(key_suffix=""):
    st.markdown(f"<h3 style='font-size:1.05rem;margin-bottom:12px'>Network Graph</h3>", unsafe_allow_html=True)
    st.plotly_chart(
        create_network_figure(engine.env, agents=engine.agents, color_by=color_by),
        use_container_width=True, key=f"net{key_suffix}")

    divider()

    if st.session_state.get('sim_history'):
        st.plotly_chart(create_cooperation_chart(st.session_state.sim_history),
                      use_container_width=True, key=f"coop{key_suffix}")
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;color:{C['text_dim']};
                    border:1px dashed {C['border']};border-radius:10px">
            <div style="font-weight:600;margin-bottom:4px">No data yet</div>
            <div style="font-size:0.82rem">Press Run Live or Fast Forward to begin</div>
        </div>
        """, unsafe_allow_html=True)


def _render_training(key_suffix=""):
    st.markdown(f"<h3 style='font-size:1.05rem;margin-bottom:12px'>Training Analytics</h3>", unsafe_allow_html=True)
    if st.session_state.get('sim_history'):
        df = pd.DataFrame(st.session_state.sim_history)
        if "DQN Loss" in df.columns and len(df) > 1:
            st.plotly_chart(create_training_chart(df), use_container_width=True, key=f"train{key_suffix}")
            if "Stranger %" in df.columns:
                st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;margin:12px 0 4px'>Stranger Edge Ratio</p>", unsafe_allow_html=True)
                st.line_chart(df.set_index("Step")["Stranger %"])
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;color:{C['text_dim']};
                    border:1px dashed {C['border']};border-radius:10px">
            <div style="font-weight:600">Training data appears here</div>
            <div style="font-size:0.82rem;margin-top:4px">DQN Loss / Temperature / Rewiring</div>
        </div>
        """, unsafe_allow_html=True)


def _render_profiles(key_suffix=""):
    st.markdown(f"<h3 style='margin-bottom:14px;font-size:1.05rem'>Cluster Analysis</h3>", unsafe_allow_html=True)
    cc = st.columns(6)
    cluster_data = [
        ("Coop Clusters", metrics['num_cooperator_clusters'], C["cooperator"]),
        ("Largest Coop", metrics['largest_cooperator_cluster'], C["cooperator"]),
        ("Defect Clusters", metrics['num_defector_clusters'], C["defector"]),
        ("Largest Defect", metrics['largest_defector_cluster'], C["defector"]),
        ("Strategy Entropy", f"{metrics['strategy_entropy']:.3f}", C["accent_glow"]),
        ("Avg Score", f"{metrics['avg_score']:.1f}", C["warning"]),
    ]
    for i, (label, val, color) in enumerate(cluster_data):
        cc[i].markdown(stat_card(label, val, color), unsafe_allow_html=True)

    divider()

    st.markdown(f"<h3 style='margin-bottom:14px;font-size:1.05rem'>Emergent Behavior</h3>", unsafe_allow_html=True)
    profile = engine.get_behavioral_profile()
    ep = st.columns(4)
    profile_data = [
        ("Chronic Cooperators", profile['chronic_cooperators'], C["cooperator"]),
        ("Chronic Defectors", profile['chronic_defectors'], C["defector"]),
        ("Swing Agents", profile['swing_agents'], C["warning"]),
        ("High Betrayal", profile['high_betrayal'], C["danger"]),
    ]
    for i, (label, val, color) in enumerate(profile_data):
        ep[i].markdown(stat_card(label, val, color), unsafe_allow_html=True)

    accent_bg = hex_to_rgba(C['accent'], 0.04)
    accent_border = hex_to_rgba(C['accent'], 0.12)

    st.markdown(f"""
    <div style="margin-top:14px;padding:12px 16px;background:{C['surface']};
                border:1px solid {C['border']};border-radius:8px;font-size:0.82rem">
        <span style="color:{C['text_dim']}">
            Avg Strategy: <b style="color:{C['text']}">{profile['avg_strategy_trend']:.2f}</b> |
            Avg Betrayal: <b style="color:{C['text']}">{profile['avg_betrayal_rate']:.2f}</b> |
            Avg Payoff: <b style="color:{C['text']}">{profile['avg_payoff_trend']:.2f}</b>
        </span>
    </div>
    <div style="margin-top:8px;padding:10px 14px;background:{accent_bg};
                border:1px solid {accent_border};border-radius:8px;
                color:{C['text_dim']};font-size:0.8rem;line-height:1.5">
        These archetypes emerged from learning with no pre-assignment.
        Cooperators formed in dense trust clusters; defectors appeared as isolated nodes or hub exploiters.
    </div>
    """, unsafe_allow_html=True)


def _render_personality(key_suffix=""):
    st.markdown(f"<h3 style='margin-bottom:14px;font-size:1.05rem'>OCEAN Personality Analysis</h3>", unsafe_allow_html=True)

    # Personality Archetypes
    if 'personality_archetypes' in metrics:
        archetypes = metrics['personality_archetypes']
        archetype_display = [
            ("Community Builders", archetypes.get('community_builder', 0), C["cooperator"]),
            ("Strategic Hubs", archetypes.get('strategic_hub', 0), C["accent_glow"]),
            ("Stoic Cooperators", archetypes.get('stoic_cooperator', 0), C["success"]),
            ("Social Butterflies", archetypes.get('social_butterfly', 0), C["warning"]),
            ("Paranoid Isolationists", archetypes.get('paranoid_isolationist', 0), C["defector"]),
            ("Opportunists", archetypes.get('opportunist', 0), C["danger"]),
        ]
        ac = st.columns(6)
        for i, (label, val, color) in enumerate(archetype_display):
            ac[i].markdown(stat_card(label, val, color), unsafe_allow_html=True)

    divider()

    # Radar chart: Cooperators vs Defectors personality profile
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Cooperator vs Defector OCEAN Profile</p>", unsafe_allow_html=True)
        st.plotly_chart(create_personality_radar(engine.agents),
                        use_container_width=True, key=f"radar{key_suffix}")

    with r2:
        st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Cooperation Rate by Personality</p>", unsafe_allow_html=True)
        if 'coop_by_personality' in metrics:
            st.plotly_chart(create_personality_cooperation_bars(metrics['coop_by_personality']),
                            use_container_width=True, key=f"coopbar{key_suffix}")

    divider()

    # Personality Distribution (box plots)
    st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Personality Distribution by Strategy</p>", unsafe_allow_html=True)
    st.plotly_chart(create_personality_distribution(engine.agents),
                    use_container_width=True, key=f"pdist{key_suffix}")

    # Assortativity
    if 'personality_assortativity' in metrics:
        divider()
        st.markdown(f"<p style='color:{C['text_dim']};font-size:0.82rem;font-weight:600;margin-bottom:8px'>Personality Echo Chambers (Assortativity)</p>", unsafe_allow_html=True)
        st.plotly_chart(create_assortativity_chart(metrics['personality_assortativity']),
                        use_container_width=True, key=f"assort{key_suffix}")
        st.markdown(f"""
        <div style="padding:8px 14px;background:{hex_to_rgba(C['accent'], 0.04)};
                    border:1px solid {hex_to_rgba(C['accent'], 0.12)};border-radius:8px;
                    color:{C['text_dim']};font-size:0.78rem;line-height:1.5">
            Positive values indicate personality-based echo chambers forming (similar agents clustering together).
            Rewiring drives this — agents preferentially connect to personality-similar others.
        </div>
        """, unsafe_allow_html=True)


# ── Render based on mode ──────────────────────────────────────────

if st.session_state.show_all:
    # Full scrollable dashboard — everything visible at once
    _render_network("_all")
    divider()
    _render_training("_all")
    divider()
    _render_profiles("_all")
    divider()
    _render_personality("_all")
else:
    # Tabbed view
    tab_net, tab_train, tab_profile, tab_personality = st.tabs(
        ["Network Graph", "Training Analytics", "Agent Profiles", "Personality"])
    with tab_net:
        _render_network("_tab")
    with tab_train:
        _render_training("_tab")
    with tab_profile:
        _render_profiles("_tab")
    with tab_personality:
        _render_personality("_tab")

# ── Execution Tail ────────────────────────────────────────────────
if st.session_state.is_playing:
    import time
    batch = st.session_state.get('live_batch_size', 5)
    for _ in range(batch):
        _record_step(engine.step())
    time.sleep(0.01)
    st.rerun()
