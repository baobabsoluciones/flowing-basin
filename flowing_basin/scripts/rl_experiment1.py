"""
RL Experiment 1
This script trains agents with different observation configurations
in both the normal and simplified environments.
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

ACTION = ["A1"]
GENERAL = ["G0", "G1"]
OBSERVATION = ["O1", "O12", "O121", "O2", "O22", "O221"]
REWARD = ["R1"]
TRAINING = ["T1", "T2"]
# TRAINING = ["T12", "T22"]

for action, general, obs, reward, training in product(ACTION, GENERAL, OBSERVATION, REWARD, TRAINING):
    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.collect_obs()
    rl.train()
