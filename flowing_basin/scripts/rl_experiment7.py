"""
RL Experiment 7
This script trains agents with a higher observation sight (32, 64 and 96, instead of only 16)
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A1", "A113"]
generals = ["G0", "G1"]
observations = ["O231", "O232", "O233"]
rewards = ["R1", "R22"]

for action, general, observation, reward in product(actions, generals, observations, rewards):

    training = "T1" if action == "A1" else "T3"

    rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    rl.train()
