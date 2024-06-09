"""
RL Experiment 12 (cont.)
This script trains the agents A21 using 6 dams with original and triple network sizes.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

best_agents = [
    "rl-A21G2O3R1T74",
    "rl-A21G2O3R1T748",
    "rl-A21G3O3R1T74",
    "rl-A21G3O3R1T748",
]

for agent in best_agents:

    rl = ReinforcementLearning(agent, verbose=3)
    rl.train()  # For testing: rl.train(num_timesteps=15, save_agent=False)
