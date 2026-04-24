"""
Topology of Trust — Design System
Dual-mode (Light / Dark) premium theme with floating sidebar and professional typography.
No emoji icons — uses clean text and CSS-based indicators only.
"""
import streamlit as st

# ── Dual Color Palettes ───────────────────────────────────────────

DARK = {
    "bg":           "#0a0e17",
    "bg_grad":      "#0f1629",
    "surface":      "#111827",
    "surface_alt":  "#1a2332",
    "border":       "#1e293b",
    "text":         "#e2e8f0",
    "text_dim":     "#94a3b8",
    "text_muted":   "#64748b",
    "accent":       "#6366f1",
    "accent_glow":  "#818cf8",
    "cooperator":   "#22d3ee",
    "defector":     "#f43f5e",
    "success":      "#10b981",
    "warning":      "#f59e0b",
    "danger":       "#ef4444",
    "chart_bg":     "rgba(10,14,23,0)",
    "grid":         "rgba(148,163,184,0.08)",
    "card_shadow":  "0 4px 24px rgba(0,0,0,0.25)",
    "hover_shadow": "0 8px 32px rgba(99,102,241,0.15)",
    "sidebar_bg":   "rgba(17,24,39,0.85)",
    "sidebar_blur": "20px",
}

LIGHT = {
    "bg":           "#f1f5f9",
    "bg_grad":      "#e2e8f0",
    "surface":      "#ffffff",
    "surface_alt":  "#f8fafc",
    "border":       "#cbd5e1",
    "text":         "#0f172a",
    "text_dim":     "#334155",
    "text_muted":   "#64748b",
    "accent":       "#1e293b",   # Sleek charcoal/black for professional light mode
    "accent_glow":  "#334155",
    "cooperator":   "#0284c7",
    "defector":     "#e11d48",
    "success":      "#059669",
    "warning":      "#d97706",
    "danger":       "#dc2626",
    "chart_bg":     "rgba(255,255,255,0)",
    "grid":         "rgba(15,23,42,0.08)",
    "card_shadow":  "0 2px 8px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.04)",
    "hover_shadow": "0 4px 20px rgba(30,41,59,0.12)",
    "sidebar_bg":   "rgba(255,255,255,0.75)",
    "sidebar_blur": "50px",
}

CHART_COLORS_DARK = ["#22d3ee", "#f59e0b", "#10b981", "#a78bfa", "#f43f5e", "#6366f1"]
CHART_COLORS_LIGHT = ["#0284c7", "#d97706", "#059669", "#7c3aed", "#e11d48", "#4f46e5"]


def _get_mode():
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    return st.session_state.theme_mode


def get_colors():
    return DARK if _get_mode() == 'dark' else LIGHT


def get_chart_colors():
    return CHART_COLORS_DARK if _get_mode() == 'dark' else CHART_COLORS_LIGHT


# ── Plotly Layout Builder ─────────────────────────────────────────

def get_plotly_layout():
    C = get_colors()
    return dict(
        paper_bgcolor=C["chart_bg"],
        plot_bgcolor=C["chart_bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
        xaxis=dict(
            gridcolor=C["grid"], zerolinecolor=C["grid"],
            showline=False, showticklabels=True,
            tickfont=dict(color=C["text_dim"], size=11),
        ),
        yaxis=dict(
            gridcolor=C["grid"], zerolinecolor=C["grid"],
            showline=False, showticklabels=True,
            tickfont=dict(color=C["text_dim"], size=11),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text_dim"])),
        margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor=C["surface"], font_color=C["text"]),
    )


def apply_premium_theme():
    """Inject global CSS for the active theme mode. Call once at top of every page."""
    C = get_colors()
    mode = _get_mode()

    # Sidebar background varies by mode
    if mode == 'dark':
        app_bg = f"linear-gradient(145deg, {C['bg']} 0%, {C['bg_grad']} 50%, {C['bg']} 100%)"
        sidebar_border = f"1px solid {C['border']}"
    else:
        app_bg = f"linear-gradient(145deg, {C['bg']} 0%, {C['bg_grad']} 100%)"
        sidebar_border = f"1px solid {C['border']}"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root ───────────────────────────────────── */
    :root, [data-testid="stAppViewContainer"] {{
        --primary-color: {C["accent"]};
        --background-color: {C["bg"]};
        --secondary-background-color: {C["surface"]};
        --text-color: {C["text"]};
    }}
    .stApp {{
        background: {app_bg};
        color: {C["text"]};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}

    /* ── Top Header ────────────────────────────── */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* ── Floating Sidebar ──────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {C["sidebar_bg"]} !important;
        backdrop-filter: blur({C["sidebar_blur"]});
        -webkit-backdrop-filter: blur({C["sidebar_blur"]});
        border-right: {sidebar_border};
        box-shadow: 4px 0 24px rgba(0,0,0,0.12);
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}

    /* Sidebar typography */
    section[data-testid="stSidebar"] {{
        color: {C["text"]} !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {C["text"]};
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {{
        color: {C["text_dim"]} !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {C["text"]} !important;
    }}

    /* Sidebar selectbox */
    section[data-testid="stSidebar"] [data-baseweb="select"] {{
        background: {C["surface"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 8px !important;
    }}

    /* ── Headers ────────────────────────────────── */
    h1 {{
        color: {C["text"]} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}
    h2, h3 {{
        color: {C["text"]} !important;
        font-weight: 600 !important;
    }}

    /* ── Metric Cards ──────────────────────────── */
    [data-testid="stMetric"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 14px 18px;
        transition: all 0.2s ease;
        box-shadow: {C["card_shadow"]};
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {C["accent"]};
        box-shadow: {C["hover_shadow"]};
        transform: translateY(-1px);
    }}
    [data-testid="stMetric"] label {{
        color: {C["text_dim"]} !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 500;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {C["text"]} !important;
        font-weight: 700;
        font-size: 1.5rem !important;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* ── Buttons ────────────────────────────────── */
    .stButton > button {{
        background: {C["surface"]};
        color: {C["text"]} !important;
        border: 1px solid {C["border"]};
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .stButton > button:hover {{
        border-color: {C["accent"]};
        color: {C["accent"]} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px {C["accent"]}22;
        background: {C["surface_alt"]};
    }}

    /* ── Tabs ───────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {C["surface"]};
        border-radius: 10px;
        padding: 4px;
        border: 1px solid {C["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px;
        color: {C["text_dim"]};
        font-weight: 500;
        font-size: 0.85rem;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C["accent"]} !important;
        color: white !important;
    }}

    /* ── Expanders ──────────────────────────────── */
    [data-testid="stExpander"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
    }}
    .streamlit-expanderHeader {{
        color: {C["text"]} !important;
        font-weight: 500;
    }}

    /* ── Toggles ────────────────────────────────── */
    [data-testid="stCheckbox"] {{
        background: {C["surface_alt"]};
        border: 1px solid {C["border"]};
        border-radius: 8px;
        padding: 8px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }}

    /* ── Progress Bar ──────────────────────────── */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {C["accent"]}, {C["cooperator"]}) !important;
        border-radius: 4px;
    }}

    /* ── Alerts ─────────────────────────────────── */
    .stAlert {{
        background: {C["surface"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 8px !important;
    }}

    /* ── Plotly containers ──────────────────────── */
    [data-testid="stPlotlyChart"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 6px;
    }}

    /* ── Number input ──────────────────────────── */
    .stNumberInput input {{
        background: {C["surface"]} !important;
        color: {C["text"]} !important;
        border: 1px solid {C["border"]} !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* ── Captions ──────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {C["text_muted"]} !important;
    }}

    /* ── Dataframe ─────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {C["border"]};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ── Scrollbar ─────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {C["border"]}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {C["accent"]}; }}

    /* ── Line chart container (st.line_chart) ─── */
    [data-testid="stVegaLiteChart"] {{
        background: {C["surface"]};
        border: 1px solid {C["border"]};
        border-radius: 10px;
        padding: 6px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_mode_toggle():
    """Renders a light/dark mode toggle in the sidebar. Call after apply_premium_theme."""
    mode = _get_mode()
    is_dark = mode == 'dark'
    
    st.sidebar.markdown(f"<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    
    # Clean professional toggle, no emojis
    new_is_dark = st.sidebar.toggle("Dark Mode", value=is_dark, key="theme_toggle_widget")
    
    if new_is_dark != is_dark:
        st.session_state.theme_mode = 'dark' if new_is_dark else 'light'
        st.rerun()


def styled_header(title, subtitle=None):
    """Renders a clean header with optional subtitle. No emojis."""
    C = get_colors()
    sub_html = f'<p style="color:{C["text_dim"]};font-size:0.95rem;margin:4px 0 0 0;font-weight:400">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <h1 style="margin:0;font-size:1.8rem;color:{C['text']}">{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def stat_card(label, value, color=None):
    """Renders a custom stat card with accent border."""
    C = get_colors()
    col = color or C["accent"]
    return f"""
    <div style="
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-left: 3px solid {col};
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: {C['card_shadow']};
        transition: all 0.2s ease;
    ">
        <div style="color:{C['text_dim']};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;font-weight:500;margin-bottom:6px">{label}</div>
        <div style="color:{C['text']};font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace">{value}</div>
    </div>
    """


def divider():
    """Renders a subtle gradient divider."""
    C = get_colors()
    st.markdown(f"""
    <div style="
        height: 1px;
        background: linear-gradient(90deg, transparent, {C['border']}, transparent);
        margin: 20px 0;
    "></div>
    """, unsafe_allow_html=True)


def section_label(text):
    """Renders a sidebar section label — clean, no emojis."""
    C = get_colors()
    st.sidebar.markdown(f"""
    <p style="
        color:{C['accent_glow']};
        font-weight:600;
        font-size:0.75rem;
        text-transform:uppercase;
        letter-spacing:0.06em;
        margin:20px 0 6px 0;
        padding-bottom:4px;
        border-bottom:1px solid {C['border']};
    ">{text}</p>
    """, unsafe_allow_html=True)
