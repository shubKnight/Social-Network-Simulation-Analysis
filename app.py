import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from engine import SimulationEngine
from visualization import draw_network

st.set_page_config(layout="wide", page_title="The Topology of Trust")

st.title("The Topology of Trust: MARL Simulation")
st.markdown("A multi-agent reinforcement learning simulation exploring the evolution of cooperation in small-world networks.")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
st.sidebar.markdown("Adjust these sliders and click **Reset Simulation** to apply.")
n = st.sidebar.slider("Number of Nodes (N)", 10, 200, 50, 10, help="Total population of the network.")
k = st.sidebar.slider("Neighbors (K)", 2, 20, 4, 2, help="Degree of each node (number of immediate connections).")
p = st.sidebar.slider("Randomness (p)", 0.0, 1.0, 0.1, 0.05, help="0 = Regular Lattice (The Village), 1 = Random Graph (The City).")
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 0.5, 0.1, 0.05, help="Probability that an agent will explore a random strategy instead of using its Q-table.")
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01, help="If agents get stuck at 50%, try a higher learning rate so they adapt to their neighbors faster.")
gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 0.9, 0.05, help="0 = agents care only about immediate reward, 0.9 = long-term reward.")

if st.sidebar.button("Reset Simulation"):
    st.session_state.engine = SimulationEngine(n, k, p, alpha=alpha, gamma=gamma, epsilon=epsilon)
    st.session_state.history = []
    st.session_state.step_count = 0

if 'engine' not in st.session_state:
    st.session_state.engine = SimulationEngine(n, k, p, alpha=alpha, gamma=gamma, epsilon=epsilon)
    st.session_state.history = []
    st.session_state.step_count = 0

engine = st.session_state.engine

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Heatmap")
    st.markdown("**Blue**: Cooperator | **Red**: Defector")
    # Draw graph
    fig = draw_network(engine.env)
    st.pyplot(fig)
    
with col2:
    st.subheader("Controls & Stats")
    
    num_steps = st.number_input("Steps to advance per click", min_value=1, max_value=5000, value=100)
    if st.button("Run Simulation (Play)", type="primary"):
        progress_text = "Running simulation steps..."
        my_bar = st.progress(0, text=progress_text)
        
        for i in range(num_steps):
            rate, _ = engine.step()
            st.session_state.step_count += 1
            st.session_state.history.append({"Step": st.session_state.step_count, "Cooperation Rate": rate})
            
            if i % max(1, num_steps // 100) == 0:
                my_bar.progress((i + 1) / num_steps, text=progress_text)
                
        my_bar.empty()
        st.rerun()
            
    st.divider()
    
    st.metric("Current Step", st.session_state.step_count)
    
    current_rate = engine.env.get_cooperation_rate()
    st.metric("Cooperation Rate", f"{current_rate:.1%}")

st.divider()

if st.session_state.history:
    st.subheader("Global Cooperation Rate Over Time")
    df = pd.DataFrame(st.session_state.history)
    st.line_chart(df.set_index("Step")["Cooperation Rate"])
