"""
RL Experiment 10
This script trains agents with discrete actions
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

generals = ["G0", "G1"]
actions = ["A31", "A32", "A33"]
rewards = ["R1", "R22"]
training = "T1002"  # Testing: T1302
observation = "O231"

for action, general, reward in product(actions, generals, rewards):

    rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    rl.train()

# Note we will train 2(G) * 3(A) * 2(R) * 1(O) * 1(T) = 12 agents of single-action blocks,
# Approximate training time: 12 agents * 0.5 hours/agent = 6 hours
