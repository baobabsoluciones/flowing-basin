"""
RL Experiment 13
This script trains agents with the adjustments action (A21) and a new version (A25).
It considers only the real environment with 2 and 6 dams (G0 and G2).
Every agent is trained for 3 replications.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

NUM_REPLICATIONS = 3

# In the case of rl-A1G0O2R1T1, rl-A1G2O2R1T14, rl-A21G0O3R1T3 and rl-A21G2O3R1T74,
# we re-train the first iteration because of incorrect training data
normal_2dams = [f"rl-A1G0O2R1T1-{i}" for i in range(NUM_REPLICATIONS)]
normal_6dams = [f"rl-A1G2O2R1T14-{i}" for i in range(NUM_REPLICATIONS) if i > 0]
adjustments_2dams = [f"rl-A21G0O3R1T3-{i}" for i in range(NUM_REPLICATIONS)]
adjustments_new_2dams = [f"rl-A25G0O3R1T3-{i}" for i in range(NUM_REPLICATIONS)]
adjustments_6dams = [f"rl-A21G2O3R1T74-{i}" for i in range(NUM_REPLICATIONS)]
adjustments_new_6dams = [f"rl-A25G2O3R1T74-{i}" for i in range(NUM_REPLICATIONS)]
discrete_2dams = [f"rl-A31G0O231R1T1002-{i}" for i in range(NUM_REPLICATIONS) if i > 0]
discrete_6dams = [f"rl-A31G2O231R1T1402-{i}" for i in range(NUM_REPLICATIONS) if i > 0]
all_agents = [
    # *normal_2dams, *normal_6dams, *adjustments_2dams, *adjustments_new_2dams, *adjustments_6dams, *adjustments_new_6dams,
    *discrete_2dams, *discrete_6dams
]
print(all_agents)
for agent in all_agents:
    rl = ReinforcementLearning(agent, verbose=3)
    # rl.train()  # For testing: rl.train(num_timesteps=15, save_agent=False)
    rl.train(num_timesteps=15, save_agent=False)
