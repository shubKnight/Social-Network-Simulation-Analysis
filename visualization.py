"""
Topology of Trust — Visualization Module
Adaptive charts that respond to light/dark theme.
Includes personality-aware visualizations for OCEAN model.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from theme import get_colors, get_plotly_layout, get_chart_colors
from agent import OCEAN_DIMS


def _base_layout(**overrides):
    """Merge current theme's Plotly layout with overrides."""
    layout = dict(get_plotly_layout())
    for k, v in overrides.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k] = layout[k].copy()
            layout[k].update(v)
        else:
            layout[k] = v
    return layout

def hex_to_rgba(hex_color, alpha):
    """Convert hex to rgba for Plotly."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


# ── Personality Color Scales ──────────────────────────────────────

OCEAN_COLORSCALES = {
    'openness':          [[0, '#1e3a5f'], [0.5, '#4a90d9'], [1, '#7ec8e3']],
    'agreeableness':     [[0, '#5c1a1a'], [0.5, '#d4a373'], [1, '#a7c957']],
    'conscientiousness': [[0, '#4a1942'], [0.5, '#8e6aac'], [1, '#c9ada7']],
    'extraversion':      [[0, '#1a3c34'], [0.5, '#f4a261'], [1, '#e76f51']],
    'neuroticism':       [[0, '#2d6a4f'], [0.5, '#e9c46a'], [1, '#e63946']],
}

OCEAN_LABELS = {
    'openness':          'Openness',
    'agreeableness':     'Agreeableness',
    'conscientiousness': 'Conscientiousness',
    'extraversion':      'Extraversion',
    'neuroticism':       'Neuroticism',
}


def create_network_figure(env, title="", agents=None, color_by="strategy"):
    """
    Interactive Plotly network graph.
    
    color_by options:
      - "strategy" (default): Cyan = Cooperator, Rose = Defector
      - Any OCEAN dimension name: continuous gradient coloring
    """
    C = get_colors()
    G = env.graph

    # Layout computation
    if env.graph_type == "grid":
        pos = {}
        side = int(np.ceil(np.sqrt(len(G.nodes()))))
        for i, node in enumerate(sorted(G.nodes())):
            pos[node] = (i % side, -(i // side))
    elif env.p < 0.15 and env.graph_type == "watts_strogatz":
        pos = nx.circular_layout(G)
    else:
        if hasattr(env, 'layout_pos') and getattr(env, 'layout_pos') is not None:
            pos = nx.spring_layout(G, pos=env.layout_pos, iterations=2)
        else:
            pos = nx.spring_layout(G, seed=42, iterations=50)
        env.layout_pos = pos

    # Edge traces
    local_x, local_y = [], []
    random_x, random_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_type = G.edges[u, v].get('edge_type', 'local')
        if edge_type == 'random':
            random_x += [x0, x1, None]
            random_y += [y0, y1, None]
        else:
            local_x += [x0, x1, None]
            local_y += [y0, y1, None]

    traces = []

    if local_x:
        traces.append(go.Scattergl(
            x=local_x, y=local_y,
            line=dict(width=1.2, color=hex_to_rgba(C['text_muted'], 0.45)),
            hoverinfo='none', mode='lines', showlegend=False,
        ))
    if random_x:
        traces.append(go.Scattergl(
            x=random_x, y=random_y,
            line=dict(width=1.0, color=hex_to_rgba(C['defector'], 0.45), dash='dot'),
            hoverinfo='none', mode='lines', showlegend=False,
        ))

    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors, node_borders, node_text, node_sizes = [], [], [], []

    scores = [G.nodes[n].get('score', 0) for n in G.nodes()]
    max_score = max(scores) if scores and max(scores) > 0 else 1

    color_by_personality = color_by in OCEAN_DIMS and agents is not None
    personality_values = []

    for node in G.nodes():
        state = G.nodes[node]['state']
        score = G.nodes[node].get('score', 0)

        if color_by_personality:
            pval = agents[node].personality[color_by]
            personality_values.append(pval)
            # We'll use a colorscale, so set placeholder colors
            node_colors.append(pval)
            node_borders.append(hex_to_rgba(C['text_muted'], 0.3))
        else:
            if state == 1:
                node_colors.append(C["cooperator"])
                node_borders.append(hex_to_rgba(C['cooperator'], 0.4))
            else:
                node_colors.append(C["defector"])
                node_borders.append(hex_to_rgba(C['defector'], 0.4))

        base_size = 7
        size_bonus = (score / max_score) * 14 if max_score > 0 else 0
        node_sizes.append(base_size + size_bonus)

        # Build tooltip with personality info
        strategy = "Cooperator" if state == 1 else "Defector"
        deg = len(list(G.neighbors(node)))
        tooltip = (
            f"<b>Node {node}</b><br>"
            f"Strategy: {strategy}<br>"
            f"Score: {score:.1f}<br>"
            f"Degree: {deg}"
        )
        if agents is not None and node in agents:
            ag = agents[node]
            tooltip += (
                f"<br>─── Personality ───<br>"
                f"Openness: {ag.openness:.2f}<br>"
                f"Agreeableness: {ag.agreeableness:.2f}<br>"
                f"Conscientiousness: {ag.conscientiousness:.2f}<br>"
                f"Extraversion: {ag.extraversion:.2f}<br>"
                f"Neuroticism: {ag.neuroticism:.2f}"
            )
        node_text.append(tooltip)

    # Glow layer
    if not color_by_personality:
        traces.append(go.Scattergl(
            x=node_x, y=node_y,
            mode='markers', hoverinfo='none', showlegend=False,
            marker=dict(color=node_colors, size=[s * 2 for s in node_sizes], opacity=0.07),
        ))

    # Main nodes
    if color_by_personality:
        traces.append(go.Scattergl(
            x=node_x, y=node_y,
            mode='markers', hoverinfo='text', text=node_text, showlegend=False,
            marker=dict(
                color=personality_values,
                colorscale=OCEAN_COLORSCALES.get(color_by, 'Viridis'),
                size=node_sizes,
                line=dict(width=1.5, color=node_borders),
                opacity=0.95,
                showscale=True,
                colorbar=dict(
                    title=dict(text=OCEAN_LABELS.get(color_by, color_by), font=dict(size=11, color=C['text_dim'])),
                    thickness=12,
                    len=0.5,
                    tickfont=dict(size=10, color=C['text_dim']),
                ),
            )
        ))
    else:
        traces.append(go.Scattergl(
            x=node_x, y=node_y,
            mode='markers', hoverinfo='text', text=node_text, showlegend=False,
            marker=dict(
                color=node_colors, size=node_sizes,
                line=dict(width=1.5, color=node_borders), opacity=0.95,
            )
        ))

    fig = go.Figure(data=traces, layout=go.Layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=13, color=C["text_dim"])) if title else None,
            showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            height=700,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    ))
    return fig


def create_cooperation_chart(history):
    """Cooperation rate over time with trend line."""
    C = get_colors()
    if not history:
        return go.Figure()

    steps = [h['Step'] for h in history]
    rates = [h['Cooperation Rate'] for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=steps, y=rates, mode='lines', name='Raw',
        line=dict(color=C["cooperator"], width=1), opacity=0.25,
    ))

    if len(rates) > 10:
        window = max(5, len(rates) // 20)
        smoothed = np.convolve(rates, np.ones(window) / window, mode='valid')
        smoothed_steps = steps[window - 1:]
        fig.add_trace(go.Scattergl(
            x=smoothed_steps, y=list(smoothed), mode='lines',
            name=f'Trend (MA-{window})',
            line=dict(color=C["cooperator"], width=3),
        ))

    fig.update_layout(**_base_layout(
        xaxis_title="Step", yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.05], gridcolor=C["grid"]),
        height=380, legend=dict(x=0.7, y=0.95),
    ))
    return fig


def create_wealth_histogram(scores):
    """Wealth distribution histogram."""
    C = get_colors()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=30,
        marker_color=C["accent_glow"], opacity=0.8,
        marker_line=dict(color=C["accent"], width=1),
    ))
    fig.update_layout(**_base_layout(
        xaxis_title="Cumulative Score", yaxis_title="Agents", height=300,
    ))
    return fig


def create_phase_transition_chart(p_values, coop_rates, std_devs=None):
    """S-curve: cooperation collapse vs randomness."""
    C = get_colors()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p_values, y=coop_rates,
        fill='tozeroy', fillcolor=hex_to_rgba(C['cooperator'], 0.08),
        mode='lines+markers', name='Cooperation Rate',
        line=dict(color=C["cooperator"], width=3),
        marker=dict(size=8, color=C["cooperator"], line=dict(color=C["surface"], width=1.5)),
        error_y=dict(type='data', array=std_devs, visible=bool(std_devs),
                     color=hex_to_rgba(C['cooperator'], 0.25)) if std_devs else None,
    ))

    if len(coop_rates) > 1:
        drops = [coop_rates[i] - coop_rates[i + 1] for i in range(len(coop_rates) - 1)]
        if max(drops) > 0.02:
            max_drop_idx = np.argmax(drops)
            critical_p = (p_values[max_drop_idx] + p_values[max_drop_idx + 1]) / 2
            fig.add_vline(
                x=critical_p, line_dash="dash", line_color=C["defector"],
                annotation_text=f"Critical p ~ {critical_p:.3f}",
                annotation_font=dict(color=C["defector"], size=11),
                annotation_position="top"
            )

    fig.update_layout(**_base_layout(
        xaxis_title="Network Randomness (p)",
        yaxis_title="Final Cooperation Rate",
        yaxis=dict(range=[0, 1.05], gridcolor=C["grid"]),
        height=480,
    ))
    return fig


def create_training_chart(df):
    """Combined training stats: loss, temperature, rewiring."""
    C = get_colors()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.33, 0.33, 0.33],
        vertical_spacing=0.12,
        subplot_titles=("DQN Training Loss", "Exploration Temperature", "Rewiring Activity"),
    )

    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["DQN Loss"], mode='lines', name='Loss',
        line=dict(color=C["warning"], width=1.5), showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Temperature"], mode='lines', name='Temp',
        line=dict(color=C["accent_glow"], width=2), showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Rewiring Events"], mode='lines', name='Rewiring',
        line=dict(color=C["success"], width=1.5), fill='tozeroy',
        fillcolor=hex_to_rgba(C["success"], 0.1), showlegend=False,
    ), row=3, col=1)

    fig.update_layout(**_base_layout(height=700, showlegend=False))

    for i in range(1, 4):
        fig.update_xaxes(gridcolor=C["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["grid"], row=i, col=1)

    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=C["text_dim"], family="Inter")

    return fig


# ── Personality Visualizations ────────────────────────────────────

def create_personality_radar(agents):
    """
    Radar/spider chart comparing OCEAN personality profiles of
    cooperators (strategy_trend > 0.6) vs defectors (strategy_trend < 0.4).
    Reveals which personality traits are associated with cooperation.
    """
    C = get_colors()
    dims = list(OCEAN_LABELS.values())

    # Split agents into cooperators and defectors by trend
    coop_agents = [a for a in agents.values() if a.strategy_trend > 0.6]
    def_agents  = [a for a in agents.values() if a.strategy_trend < 0.4]

    coop_means = [np.mean([a.personality[d] for a in coop_agents]) if coop_agents else 0.5
                  for d in OCEAN_DIMS]
    def_means  = [np.mean([a.personality[d] for a in def_agents]) if def_agents else 0.5
                  for d in OCEAN_DIMS]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=coop_means + [coop_means[0]],
        theta=dims + [dims[0]],
        fill='toself',
        fillcolor=hex_to_rgba(C['cooperator'], 0.12),
        line=dict(color=C['cooperator'], width=2.5),
        name=f'Cooperators ({len(coop_agents)})',
    ))

    fig.add_trace(go.Scatterpolar(
        r=def_means + [def_means[0]],
        theta=dims + [dims[0]],
        fill='toself',
        fillcolor=hex_to_rgba(C['defector'], 0.12),
        line=dict(color=C['defector'], width=2.5),
        name=f'Defectors ({len(def_agents)})',
    ))

    fig.update_layout(
        **_base_layout(
            height=420,
            showlegend=True,
            legend=dict(x=0.02, y=1.12, orientation='h', font=dict(size=11, color=C['text_dim'])),
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=C['grid'],
                    tickfont=dict(size=9, color=C['text_muted']),
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color=C['text_dim']),
                    gridcolor=C['grid'],
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
        )
    )
    return fig


def create_personality_cooperation_bars(coop_by_personality):
    """
    Grouped bar chart showing cooperation rate by personality dimension,
    split into Low / Mid / High bins.
    """
    C = get_colors()
    CC = get_chart_colors()

    dims = [OCEAN_LABELS[d] for d in OCEAN_DIMS]
    low_vals  = [coop_by_personality[d]['low']  for d in OCEAN_DIMS]
    mid_vals  = [coop_by_personality[d]['mid']  for d in OCEAN_DIMS]
    high_vals = [coop_by_personality[d]['high'] for d in OCEAN_DIMS]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Low (<0.33)', x=dims, y=low_vals,
        marker_color=CC[4], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name='Mid (0.33-0.67)', x=dims, y=mid_vals,
        marker_color=CC[3], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name='High (>0.67)', x=dims, y=high_vals,
        marker_color=CC[0], opacity=0.85,
    ))

    fig.update_layout(
        **_base_layout(
            height=380,
            legend=dict(x=0.65, y=0.95, font=dict(size=10, color=C['text_dim'])),
            barmode='group',
            xaxis_title="Personality Dimension",
            yaxis_title="Avg Cooperation Trend",
            yaxis=dict(range=[0, 1.05], gridcolor=C['grid']),
        )
    )
    return fig


def create_personality_distribution(agents):
    """
    Violin / box plots showing the distribution of each OCEAN dimension
    across all agents, split by current strategy.
    """
    C = get_colors()

    fig = make_subplots(rows=1, cols=5, shared_yaxes=True,
                        subplot_titles=[OCEAN_LABELS[d] for d in OCEAN_DIMS])

    for i, dim in enumerate(OCEAN_DIMS, 1):
        coop_vals = [a.personality[dim] for a in agents.values() if a.strategy == 1]
        def_vals  = [a.personality[dim] for a in agents.values() if a.strategy == 0]

        fig.add_trace(go.Box(
            y=coop_vals, name='Coop', marker_color=C['cooperator'],
            showlegend=(i == 1), legendgroup='coop',
            boxmean=True, jitter=0.3, pointpos=-1.5,
        ), row=1, col=i)

        fig.add_trace(go.Box(
            y=def_vals, name='Defect', marker_color=C['defector'],
            showlegend=(i == 1), legendgroup='defect',
            boxmean=True, jitter=0.3, pointpos=1.5,
        ), row=1, col=i)

    fig.update_layout(
        **_base_layout(
            height=350,
            legend=dict(x=0.85, y=1.1, font=dict(size=10, color=C['text_dim'])),
            yaxis=dict(range=[-0.05, 1.05], gridcolor=C['grid']),
        )
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=10, color=C['text_dim'], family='Inter')

    return fig


def create_assortativity_chart(assortativity_data):
    """
    Horizontal bar chart showing personality assortativity for each OCEAN dimension.
    Positive = echo chambers forming, Negative = opposites attract.
    """
    C = get_colors()

    dims = [OCEAN_LABELS[d] for d in OCEAN_DIMS]
    vals = [assortativity_data.get(d, 0.0) for d in OCEAN_DIMS]
    colors = [C['cooperator'] if v > 0 else C['defector'] for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dims, x=vals, orientation='h',
        marker_color=colors, opacity=0.85,
        marker_line=dict(width=1, color=[hex_to_rgba(c, 0.6) for c in colors]),
    ))

    fig.add_vline(x=0, line_color=C['text_muted'], line_width=1)

    fig.update_layout(
        **_base_layout(
            height=280,
            xaxis_title="Assortativity Coefficient",
            xaxis=dict(range=[-1, 1], gridcolor=C['grid']),
            yaxis=dict(gridcolor=C['grid']),
        )
    )
    return fig
