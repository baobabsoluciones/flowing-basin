"""
RL Experiment 7
This script trains agents with a higher observation sight (32, 64 and 96, instead of only 16)
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import json

actions = ["A1", "A113"]
generals = ["G0", "G1"]
observations = ["O231", "O232", "O233"]
rewards = ["R1", "R22"]

agents = []

for action, general, observation, reward in product(actions, generals, observations, rewards):

    training = "T1" if action == "A1" else "T3"

    agents.append(f"rl-{action}{general}{observation}{reward}{training}")
    # rl = ReinforcementLearning(f"{action}{general}{observation}{reward}{training}", verbose=2)
    # rl.train()

experiment = {
    "description": """RL Experiment 7
This script trains agents with a higher observation sight (32, 64 and 96, instead of only 16)""",
    "agents": agents
}
with open('experiments/experiment7.json', 'w') as f:
    json.dump(experiment, f, indent=4)
