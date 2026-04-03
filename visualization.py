import plotly.graph_objects as go
import networkx as nx
import numpy as np


def create_network_figure(env, title="Network State"):
    """
    Creates an interactive Plotly network graph.
    Blue = Cooperator, Red = Defector.
    Node size reflects cumulative score.
    """
    G = env.graph
    
    # Compute layout based on graph type
    if env.graph_type == "grid":
        pos = {}
        side = int(np.ceil(np.sqrt(len(G.nodes()))))
        for i, node in enumerate(sorted(G.nodes())):
            pos[node] = (i % side, -(i // side))
    elif env.p < 0.15 and env.graph_type == "watts_strogatz":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, iterations=50)
    
    # Edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    
    node_colors = []
    node_text = []
    node_sizes = []
    
    scores = [G.nodes[n].get('score', 0) for n in G.nodes()]
    max_score = max(scores) if scores and max(scores) > 0 else 1
    
    for node in G.nodes():
        state = G.nodes[node]['state']
        score = G.nodes[node].get('score', 0)
        
        if state == 1:
            node_colors.append('#3b82f6')  # Blue for cooperator
        else:
            node_colors.append('#ef4444')  # Red for defector
        
        # Size proportional to score
        base_size = 8
        size_bonus = (score / max_score) * 15 if max_score > 0 else 0
        node_sizes.append(base_size + size_bonus)
        
        strategy = "Cooperator" if state == 1 else "Defector"
        node_text.append(
            f"Node {node}<br>"
            f"Strategy: {strategy}<br>"
            f"Score: {score:.1f}<br>"
            f"Neighbors: {len(list(G.neighbors(node)))}"
        )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='white'),
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    )
    
    return fig


def create_cooperation_chart(history):
    """Creates a cooperation rate over time chart with smoothed trendline."""
    if not history:
        return go.Figure()
    
    steps = [h['Step'] for h in history]
    rates = [h['Cooperation Rate'] for h in history]
    
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=steps, y=rates,
        mode='lines',
        name='Cooperation Rate',
        line=dict(color='#3b82f6', width=1),
        opacity=0.4,
    ))
    
    # Smoothed trendline (moving average)
    if len(rates) > 10:
        window = max(5, len(rates) // 20)
        smoothed = np.convolve(rates, np.ones(window) / window, mode='valid')
        smoothed_steps = steps[window - 1:]
        fig.add_trace(go.Scatter(
            x=smoothed_steps, y=smoothed,
            mode='lines',
            name=f'Trend (MA-{window})',
            line=dict(color='#2563eb', width=3),
        ))
    
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Cooperation Rate",
        yaxis=dict(range=[0, 1.05]),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(x=0.7, y=0.95),
    )
    
    return fig


def create_wealth_histogram(scores):
    """Creates a wealth distribution histogram."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=30,
        marker_color='#8b5cf6',
        opacity=0.8,
    ))
    fig.update_layout(
        xaxis_title="Cumulative Score",
        yaxis_title="Number of Agents",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def create_phase_transition_chart(p_values, coop_rates, std_devs=None):
    """Creates the classic S-curve showing cooperation collapse vs randomness."""
    fig = go.Figure()
    
    if std_devs:
        fig.add_trace(go.Scatter(
            x=p_values, y=coop_rates,
            error_y=dict(type='data', array=std_devs, visible=True, color='rgba(59,130,246,0.3)'),
            mode='lines+markers',
            name='Avg Cooperation Rate',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=p_values, y=coop_rates,
            mode='lines+markers',
            name='Cooperation Rate',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8),
        ))
    
    # Find the tipping point (biggest drop)
    if len(coop_rates) > 1:
        drops = [coop_rates[i] - coop_rates[i + 1] for i in range(len(coop_rates) - 1)]
        max_drop_idx = np.argmax(drops)
        critical_p = (p_values[max_drop_idx] + p_values[max_drop_idx + 1]) / 2
        
        fig.add_vline(
            x=critical_p,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Critical Threshold ≈ {critical_p:.3f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        xaxis_title="Network Randomness (p)",
        yaxis_title="Final Cooperation Rate",
        yaxis=dict(range=[0, 1.05]),
        height=450,
        margin=dict(l=50, r=20, t=30, b=50),
    )
    
    return fig
