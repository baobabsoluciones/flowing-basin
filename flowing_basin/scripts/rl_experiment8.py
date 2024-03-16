"""
RL Experiment 8
This script trains agents with reward adjusted according to MILP estimated performance
and/or with rl-greedy's reference recomputed for every period
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A1", "A113"]
generals = ["G0", "G1"]
observations = ["O231", "O232"]

for action, general, observation in product(actions, generals, observations):

    # Recomputing rl-greedy's average reward only makes a difference when there is more than one step per episode
    rewards = ["R221", "R222", "R223", "R23", "R231", "R232", "R233"] if action == "A1" else ["R23"]

    for reward in rewards:

        training = "T1" if action == "A1" else "T3"

        rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
        rl.train()

# Note we will train 2(G) * 2(O) * 7(R) + 2(G) * 2(O) = 28 + 4 = 32 agents
