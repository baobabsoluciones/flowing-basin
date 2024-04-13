"""
RL Experiment 10
This script trains agents with discrete actions
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product

generals = ["G0", "G1"]
actions = ["A1", "A31", "A32", "A33", "A313", "A323", "A333"]
multi_block = {"A313", "A323", "A333"}

for action, general in product(actions, generals):

    # Train with A2C and PPO
    # Multi-action agents have a larger network
    # Also, given their high training time, multi-action agents are only trained with reward R22
    if action in multi_block:
        trainings = ["T1071", "T1072"]
        observation = "O2"
        rewards = ["R22"]
    else:
        trainings = ["T1001", "T1002"]
        observation = "O231"
        rewards = ["R1", "R22"]

    for training, reward in product(trainings, rewards):
        rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
        rl.train()

# Note we will train 4(A) * 2(G) * 1(O) * 2(R) * 2(T) = 32 agents of single-action blocks,
# and 3(A) * 2(G) * 1(O) * 1(R) * 2(T) = 12 agents of multi-action blocks
# Approximate training time: 32 agents * 0.5 hours/agent + 12 agents * 6.5 hours/agent = 94 hours = 4 days
