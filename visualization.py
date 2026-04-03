import matplotlib.pyplot as plt
import networkx as nx

def draw_network(env):
    """
    Draws the network with nodes colored by strategy.
    Blue = Cooperator, Red = Defector
    """
    G = env.graph
    
    color_map = []
    for node in G.nodes():
        if G.nodes[node]['state'] == 1:
            color_map.append('#3b82f6') # Modern blue
        else:
            color_map.append('#ef4444') # Modern red
            
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # We use a spring layout or circular layout. 
    # For Watts-Strogatz with low p, circular looks better. Spring is better for high p.
    if env.p < 0.2:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
        
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=150, ax=ax, edge_color='#d1d5db', linewidths=0.5)
    
    fig.patch.set_facecolor('#ffffff') # White background
    ax.axis('off')
    
    return fig
