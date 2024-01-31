"""
RL Experiment 6
This script trains agents with the "adjustments" action type ("A21", "A22", "A23", and "A24")
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

actions = ["A21", "A22", "A23", "A24"]
generals = ["G0", "G1"]
obs = "O3"
reward = "R1"
training = "T3"

for action, general in product(actions, generals):

    # Skip agent that was already trained successfully before
    if action == "A21" and general == "G0":
        continue

    rl = ReinforcementLearning(f"{action}{general}{obs}{reward}{training}", verbose=2)
    rl.train()
