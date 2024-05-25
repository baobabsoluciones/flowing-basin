"""
RL Experiment 12
This script trains the best agents of previous experiments using 6 dams instead of 2.

The best agents are:
- For G0: rl-A31G0O231R1T1002 from Experiment 10 (+4.25% above rl-greedy) & rl-A1G0O231R22T1 from Experiment 7 (+3.01%)
& rl-A1G0O2R1T1 from Experiment 1 (+2.21%).
- For G1: rl-A113G1O2R22T302 from Experiment 9 (+1.54% above rl-greedy), rl-A113G1O232R22T3 from Experiment 7 (+1.53%)
& rl-A23G1O3R1T3 from Experiment 6 (+1.25%).
However, we will also try the agents rl-A31G0O231R1T1002, rl-A1G0O231R22T1 & rl-A1G0O2R1T1 in G1,
and we will also try the agent rl-A23G1O3R1T3 in G0.

To train these agents for 6 dams:
- G0 and G1 will be replaced by G2 and G3, respectively.
- We will double the training timesteps (by using T.4.. instead of T.0..).
- With A113 and A23 actions, we will train the agents with an even smaller replay buffer
(by using T7... instead of T3...) to avoid an _ArrayMemoryError.
- With A113 and A23 actions, we will train both the original agents and those with a triple network size.

We do not try tripling the size of networks with A1 or A31 actions because these observation spaces
do not increase so much. This is because we exclude future_inflows in dam3, ..., dam6 since these will always be 0.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

# We comment out those agents already trained
best_agents = [
    # "rl-A31G2O231R1T1402",
    # "rl-A31G3O231R1T1402",
    # "rl-A1G2O231R22T14",
    # "rl-A1G3O231R22T14",
    "rl-A1G2O2R1T14",
    "rl-A1G3O2R1T14",
    "rl-A23G2O3R1T74",
    "rl-A23G2O3R1T748",
    "rl-A23G3O3R1T74",
    "rl-A23G3O3R1T748",
    # "rl-A113G3O2R22T342",
    "rl-A113G3O2R22T749",
    "rl-A113G3O232R22T74",
    "rl-A113G3O232R22T748"
]

for agent in best_agents:

    rl = ReinforcementLearning(agent, verbose=3)
    rl.train()  # For testing: rl.train(num_timesteps=15, save_agent=False)
