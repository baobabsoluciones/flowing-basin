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
TRAINING = ["T1"]
# TRAINING = ["T12"]

for action, general, obs, reward, training in product(ACTION, GENERAL, OBSERVATION, REWARD, TRAINING):
    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()
