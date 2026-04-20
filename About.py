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
    ">The Topology of Trust</h1>
    <p style="
        color:{C['text_dim']};
        font-size:1.05rem;
        max-width:600px;
        margin:0 auto;
        line-height:1.7;
    ">
        Graph Convolutional Multi-Agent Reinforcement Learning<br>
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
        Does the structure of a social network determine whether trust can survive?
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Cards ─────────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>Experiment Modules</h2>", unsafe_allow_html=True)

nav_items = [
    ("Simulation", "Run the live simulation. Watch agents cooperate and defect in real-time on an interactive network graph.", C["cooperator"]),
    ("Phase Transition", "Automated sweep of network randomness to find the exact tipping point where trust collapses.", C["warning"]),
    ("Network Compare", "Compare how trust evolves on different network types: Small-World, Scale-Free, Random, and Grid.", C["accent_glow"]),
    ("Resilience Lab", "Shock a stable cooperative society by injecting defectors. Can the network recover?", C["defector"]),
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
            min-height:180px;
            box-shadow:{C['card_shadow']};
        ">
            <div style="color:{C['text']};font-weight:700;font-size:0.95rem;margin-bottom:10px">{title}</div>
            <div style="color:{C['text_dim']};font-size:0.8rem;line-height:1.55">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────
divider()

st.markdown(f"<h2 style='text-align:center;margin-bottom:24px;font-size:1.3rem'>How It Works</h2>", unsafe_allow_html=True)

how_items = [
    ("100 AI agents live on graph nodes, each powered by a shared Graph Convolutional Network (GCN) that reads the entire network topology.", C["success"]),
    ("Edges represent social connections. Agents interact with their neighbours via the Iterated Prisoner's Dilemma — cooperate or betray.", C["accent_glow"]),
    ("The GCN learns Q-values for every node simultaneously via Deep Q-Learning with experience replay and target network stabilisation.", C["cooperator"]),
    ("Exploited cooperators dynamically rewire — cutting untrustworthy defectors and seeking reputable partners in their 2-hop neighbourhood.", C["warning"]),
    ("Topology matters. In local networks, cooperators form defensive clusters. As randomness increases (like social media), trust erodes.", C["defector"]),
]

for text, color in how_items:
    st.markdown(f"""
    <div style="
        display:flex;align-items:flex-start;gap:14px;
        padding:14px 18px;margin:6px 0;
        background:{C['surface']};
        border-left:3px solid {color};
        border-radius:8px;
    ">
        <div style="
            width:6px;height:6px;min-width:6px;
            background:{color};border-radius:50%;
            margin-top:7px;
        "></div>
        <span style="color:{C['text']};font-size:0.88rem;line-height:1.6">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
divider()

st.markdown(f"""
<div style="text-align:center;padding:20px">
    <p style="color:{C['text_dim']};font-size:1.05rem;font-style:italic;font-weight:500;margin-bottom:8px">
        "A few global shortcuts can destroy centuries of locally-built trust."
    </p>
    <p style="color:{C['text_muted']};font-size:0.78rem">
        Python · PyTorch · NetworkX · Streamlit · Plotly
    </p>
</div>
""", unsafe_allow_html=True)
