"""
RL Experiment 3
This script trains agents with differently sized blocks of actions
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

ACTION = ["A1", "A110", "A111", "A112", "A113"]
GENERAL = ["G0", "G1"]
OBSERVATION = ["O2"]
REWARD = ["R1"]

for action, general, obs, reward in product(ACTION, GENERAL, OBSERVATION, REWARD):

    # Use a smaller replay buffer when action is A113, which has too big observation arrays
    training = "T1" if action != "A113" else "T3"

    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()
