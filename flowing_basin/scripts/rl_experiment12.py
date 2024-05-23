"""
RL Experiment 12
This script trains the best agents of previous experiments using 6 dams instead of 2.

The best agents are:
- For G0: rl-A31G0O231R1T1002 from Experiment 10 (+4.25% above rl-greedy) & rl-A1G0O231R22T1 from Experiment 7 (+3.01%).
- For G1: rl-A113G1O2R22T302 from Experiment 9 (+1.54% above rl-greedy) & rl-A113G1O232R22T3 from Experiment 7 (+1.53%).
However, we will also try the agents rl-A31G0O231R1T1002 and rl-A1G0O231R22T1 in G1.

To train these agents for 6 dams:
- G0 and G1 will be replaced by G2 and G3, respectively.
- We will double the training timesteps (by using T.4.. instead of T.0..).
- With A113 actions, we will train both the original agents and those with a triple network size.

We do not try tripling the size of networks with A1 or A31 actions because these observation spaces
do not increase so much. This is because we exclude future_inflows in dam3, ..., dam6 since these will always be 0.
"""

from flowing_basin.solvers.rl import ReinforcementLearning

best_agents = [
    "rl-A31G2O231R1T1402",
    "rl-A31G3O231R1T1402",
    "rl-A1G2O231R22T14",
    "rl-A1G3O231R22T14",
    "rl-A113G3O2R22T342",
    "rl-A113G3O2R22T349",
    "rl-A113G3O232R22T34",
    "rl-A113G3O232R22T3480"
]

for agent in best_agents:

    rl = ReinforcementLearning(agent, verbose=3)
    rl.train(num_timesteps=15, save_agent=False)
