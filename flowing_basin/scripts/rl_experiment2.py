"""
RL Experiment 2
This script trains agents with normal and randomized features to
check if the agents make use of the observation
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

ACTION = ["A1"]
GENERAL = ["G0", "G1"]
OBSERVATION = ["O2", "O20", "O201"]
REWARD = ["R1"]
TRAINING = ["T1"]
# TRAINING = ["T12"]

for action, general, obs, reward, training in product(ACTION, GENERAL, OBSERVATION, REWARD, TRAINING):
    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()
