"""
Topology of Trust — Visualization Module
Adaptive charts that respond to light/dark theme.
"""
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from theme import get_colors, get_plotly_layout, get_chart_colors


def _base_layout(**overrides):
    """Merge current theme's Plotly layout with overrides."""
    layout = dict(get_plotly_layout())
    layout.update(overrides)
    return layout

def hex_to_rgba(hex_color, alpha):
    """Convert hex to rgba for Plotly."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def create_network_figure(env, title=""):
    """
    Interactive Plotly network graph.
    Cyan/teal = Cooperator, Rose/red = Defector.
    Separate edge traces for local vs random connections.
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
            # If layout exists, only do 2 iterations from the previous state to settle any rewired edges
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

    for node in G.nodes():
        state = G.nodes[node]['state']
        score = G.nodes[node].get('score', 0)

        if state == 1:
            node_colors.append(C["cooperator"])
            node_borders.append(hex_to_rgba(C['cooperator'], 0.4))
        else:
            node_colors.append(C["defector"])
            node_borders.append(hex_to_rgba(C['defector'], 0.4))

        base_size = 7
        size_bonus = (score / max_score) * 14 if max_score > 0 else 0
        node_sizes.append(base_size + size_bonus)

        strategy = "Cooperator" if state == 1 else "Defector"
        deg = len(list(G.neighbors(node)))
        node_text.append(
            f"<b>Node {node}</b><br>"
            f"Strategy: {strategy}<br>"
            f"Score: {score:.1f}<br>"
            f"Degree: {deg}"
        )

    # Glow layer
    traces.append(go.Scattergl(
        x=node_x, y=node_y,
        mode='markers', hoverinfo='none', showlegend=False,
        marker=dict(color=node_colors, size=[s * 2 for s in node_sizes], opacity=0.07),
    ))

    # Main nodes
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
    from plotly.subplots import make_subplots
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

    fig.update_layout(**_base_layout(height=700), showlegend=False)

    for i in range(1, 4):
        fig.update_xaxes(gridcolor=C["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["grid"], row=i, col=1)

    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=C["text_dim"], family="Inter")

    return fig
