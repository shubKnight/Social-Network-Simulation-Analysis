Hey everyone, I’m building a project for my university Machine Learning course called **"Social network analysis using iterated game theory"** and I've hit a wall.  

What I'm doing and how it should work:  
The goal of the project is to simulate how real human societies build trust. I have 100 agents playing the Iterated Prisoner's Dilemma, placed on different complex network topologies (like Watts-Strogatz "villages" vs. Barabási-Albert "cities"). Initially I thought of it as an RL project but don’t know what to do next.  

The theoretical math says that highly clustered networks (villages) should naturally protect cooperators because they form tight-knit communities that block out defectors. The simulation should mimic how actual human society behaves and organizes itself to survive.  

## My Flow of Implementation:
I've gone through several iterations to try and capture this realistic behavior:  

- **Basic Agent Simulation:** Started with simple isolated Reinforcement Learning agents.  
- **MARL & Q-Learning:** Upgraded to Multi-Agent RL using standard Q-table learning and the Bellman equation.  
- **Spatial Awareness:** Realizing they lacked structural context, I tried feeding them local neighborhood spatial features.  
- **Evolutionary Game Theory (EGT):** Briefly pivoted to pure EGT (agents imitating successful neighbors). It worked a bit better but wasn't giving me the perfect results I expected, plus I really need to use Machine Learning algorithms for this project course requirement.  
- **Deep Q-Learning (DQL):** I shifted back to ML and implemented a Deep Q-Network, hoping the Neural Net would generalize the topology better.  
- **Graph Neural Networks (GNN):** Finally, hoping to definitively solve this by giving the Neural Network full topological context, I built a custom Graph Convolutional Network (GCN) PyTorch brain. The network takes the environment's Adjacency Matrix and computes Q-values directly over the graph topology. I even added "adaptive rewiring" where cooperators can sever ties with defectors. I thought GNNs would be the ultimate solution... but it ended in disappointment.  

## The Issue:
Despite using a GNN directly on the Adjacency matrix, instead of acting like an actual society where localized trust clusters naturally form and defend themselves, the simulation is completely unstable. The agents either globally lock into 100% cooperation or suddenly crash to 0%, ignoring the topology. The deep RL network just doesn't naturally capture or "care" about the local cluster effects at all.  

## Please Help!
What can I do further to solve this? Am I doing something fundamentally wrong by using Q-Learning / Neural Networks for a spatial social dilemma? Are there any errors in my architectural assumptions, or should I try doing something else entirely? Any recommendations, paper links, or advice would be a lifesaver.  

Here is the GitHub link to the project:  
https://github.com/shubKnight/Social-Network-Simulation-Analysis