import streamlit as st
from theme import apply_premium_theme, get_colors, render_mode_toggle, styled_header, divider

st.set_page_config(
    layout="wide",
    page_title="Topology of Trust",
    page_icon="T",
    initial_sidebar_state="collapsed",
)
apply_premium_theme()
render_mode_toggle()

C = get_colors()

# ── Hero Section ──────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:50px 20px 36px;position:relative">
    <div style="
        position:absolute;top:50%;left:50%;
        width:350px;height:350px;
        background:radial-gradient(circle, {C['accent']}12 0%, transparent 70%);
        transform:translate(-50%,-50%);
        border-radius:50%;pointer-events:none;
    "></div>
    <h1 style="
        font-size:2.8rem !important;
        font-weight:900;
        letter-spacing:-0.03em;
        margin:0 0 8px 0;
        color:{C['text']} !important;
    ">Topology of Trust</h1>
    <p style="
        color:{C['text_dim']};
        font-size:1.05rem;
        max-width:640px;
        margin:0 auto;
        line-height:1.7;
    ">
        Graph Convolutional Multi-Agent Reinforcement Learning</br>
        for Modelling Cooperation Dynamics on Social Networks
    </p>
</div>
""", unsafe_allow_html=True)

divider()

# ── Central Question ──────────────────────────────────────────────
st.markdown(f"""
<div style="
    text-align:center;
    padding:28px;
    margin:16px auto;
    max-width:640px;
    background:{C['surface']};
    border:1px solid {C['border']};
    border-radius:14px;
    box-shadow:{C['card_shadow']};
">
    <p style="color:{C['text_dim']};font-size:0.82rem;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin:0 0 10px 0">
        Research Question
    </p>
    <p style="color:{C['accent_glow']};font-size:1.35rem;font-weight:700;margin:0;font-style:italic">
        How do individual personality and network structure interact to determine whether cooperation can survive?
    </p>
</div>
""", unsafe_allow_html=True)

# ── What This Is ──────────────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>About the Simulation</h2>", unsafe_allow_html=True)

overview_text = f"""
<div style="
    max-width:760px;margin:0 auto;
    padding:22px 26px;
    background:{C['surface']};
    border:1px solid {C['border']};
    border-radius:12px;
    box-shadow:{C['card_shadow']};
    color:{C['text']};
    font-size:0.88rem;
    line-height:1.75;
">
    <p style="margin:0 0 14px 0">
        This framework simulates the emergence of cooperation and defection in social networks.
        Each agent is governed by a <b>shared Graph Convolutional Network</b> (GCN) that performs
        3-hop neighbourhood aggregation — allowing agents to perceive cooperation patterns
        across their extended social graph — and an intrinsic
        <b>OCEAN personality profile</b> drawn from psychology's Big Five model.
    </p>
    <p style="margin:0 0 14px 0">
        Agents interact through the Iterated Prisoner's Dilemma, rewire their connections
        based on personality homophily, and learn strategies through Deep Q-Learning.
        Personality traits drift dynamically based on social experiences: betrayal raises
        neuroticism, mutual cooperation builds agreeableness, and exposure to strangers
        modulates openness.
    </p>
    <p style="margin:0">
        The result is a self-organising system where personality-driven echo chambers,
        trust clusters, and defection cascades emerge without any hardcoded behavioural rules.
    </p>
</div>
"""
st.markdown(overview_text, unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>Technical Architecture</h2>", unsafe_allow_html=True)

arch_items = [
    ("Graph Convolutional Network",
     "A 3-layer GCN reads the full network adjacency matrix, performing message-passing so each agent's decision is informed by its 3-hop neighbourhood. An MLP head estimates Q-values for Cooperate and Defect.",
     C["cooperator"]),
    ("OCEAN Personality Model",
     "Each agent carries five continuous personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) allocated via a Dirichlet-constrained budget. These traits directly modulate exploration, discount factor, rewiring aggression, and emotional reactivity.",
     C["accent_glow"]),
    ("Dynamic Edge Trust",
     "Every connection carries a trust value between 0 and 1 that evolves with interaction. Mutual cooperation builds trust incrementally; defection shatters it. Payoffs are scaled by trust, modelling the asymmetry between slow trust-building and rapid trust destruction.",
     C["warning"]),
    ("Co-Evolutionary Rewiring",
     "Exploited agents can sever connections to chronic defectors and seek personality-similar replacements via homophily. This drives the spontaneous formation of cooperative clusters and personality-based echo chambers.",
     C["success"]),
    ("Personality Drift",
     "Traits are not static. Social experiences shift personality over time — being suckered raises neuroticism, cooperation builds agreeableness, novel encounters modulate openness. A baseline regression anchor prevents population-wide psychological collapse.",
     C["defector"]),
]

for title, desc, color in arch_items:
    st.markdown(f"""
    <div style="
        display:flex;align-items:flex-start;gap:16px;
        padding:16px 20px;margin:8px auto;
        max-width:760px;
        background:{C['surface']};
        border-left:3px solid {color};
        border-radius:8px;
        box-shadow:{C['card_shadow']};
    ">
        <div>
            <div style="color:{C['text']};font-weight:700;font-size:0.92rem;margin-bottom:6px">{title}</div>
            <div style="color:{C['text_dim']};font-size:0.82rem;line-height:1.6">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Experiment Modules ───────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>Experiment Modules</h2>", unsafe_allow_html=True)

nav_items = [
    ("Live Simulation",
     "Run the simulation in real-time. Configure network topology, payoff matrix, and RL hyperparameters. Visualise the network graph, cooperation timeline, personality radar charts, and training analytics.",
     C["cooperator"]),
    ("Phase Transition",
     "Automated sweep across network randomness to identify the critical threshold where cooperative structures collapse as the network transitions from lattice to random graph.",
     C["warning"]),
    ("Network Comparison",
     "Side-by-side comparison of how trust evolves on Small-World, Scale-Free, Random, and Grid topologies under identical parameters. Reveals the structural determinants of cooperation.",
     C["accent_glow"]),
    ("Resilience Lab",
     "Controlled shock-and-recovery experiments. Injects defectors with configurable size, frequency, and cascade depth. Measures whether the network can self-heal from targeted disruption.",
     C["defector"]),
]

cols = st.columns(4)
for i, (title, desc, color) in enumerate(nav_items):
    with cols[i]:
        st.markdown(f"""
        <div style="
            background:{C['surface']};
            border:1px solid {C['border']};
            border-top:3px solid {color};
            border-radius:12px;
            padding:22px 18px;
            min-height:200px;
            box-shadow:{C['card_shadow']};
        ">
            <div style="color:{C['text']};font-weight:700;font-size:0.95rem;margin-bottom:10px">{title}</div>
            <div style="color:{C['text_dim']};font-size:0.8rem;line-height:1.55">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>Stack</h2>", unsafe_allow_html=True)

stack_items = [
    ("Python 3.9+", "Core language"),
    ("PyTorch", "GCN-DQN, experience replay, Huber loss training"),
    ("NetworkX", "Graph generation, topological metrics, rewiring"),
    ("Streamlit", "Interactive dashboard and real-time controls"),
    ("Plotly", "Network visualisation, charts, and personality analysis"),
    ("NumPy", "Numerical computation and analytics"),
]

stack_cols = st.columns(6)
for i, (name, role) in enumerate(stack_items):
    with stack_cols[i]:
        st.markdown(f"""
        <div style="
            background:{C['surface']};
            border:1px solid {C['border']};
            border-radius:10px;
            padding:16px 12px;
            text-align:center;
            box-shadow:{C['card_shadow']};
            min-height:90px;
        ">
            <div style="color:{C['text']};font-weight:700;font-size:0.85rem;margin-bottom:6px">{name}</div>
            <div style="color:{C['text_muted']};font-size:0.72rem;line-height:1.4">{role}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
divider()

st.markdown(f"""
<div style="text-align:center;padding:20px">
    <p style="color:{C['text_dim']};font-size:1.05rem;font-style:italic;font-weight:500;margin-bottom:8px">
        "A few global shortcuts can destroy centuries of locally-built trust."
    </p>
    <p style="color:{C['text_muted']};font-size:0.72rem;margin-top:12px">
        Topology of Trust — Shubham Kumar
    </p>
</div>
""", unsafe_allow_html=True)
