"""
RL Experiment 10
This script trains agents with discrete actions
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import json

generals = ["G0", "G1"]
actions = ["A31", "A32", "A33"]
rewards = ["R1", "R22"]
training = "T1002"  # Testing: T1302
observation = "O231"

agents = []

for action, general, reward in product(actions, generals, rewards):

    agents.append(f"rl-{action}{general}{observation}{reward}{training}")
    # rl = ReinforcementLearning(, verbose=2)
    # rl.train(save_tensorboard=False)

experiment = {
    "description": """RL Experiment 10
This script trains agents with discrete actions""",
    "agents": agents
}
with open('experiments/experiment10.json', 'w') as f:
    json.dump(experiment, f, indent=4)

# Note we will train 2(G) * 3(A) * 2(R) * 1(O) * 1(T) = 12 agents of single-action blocks,
# Approximate training time: 12 agents * 0.5 hours/agent = 6 hours
